"""Pull just our movies' subtitles out of a 25GB remote archive.

The OPUS OpenSubtitles corpus ships as one 25GB ZIP64 file. We want ~5k of
its ~450k subtitle files, so downloading the whole thing to use 2% of it is
absurd — and the per-movie files aren't exposed as separate objects.

The server supports HTTP range requests, so instead this reads the zip
remotely: fetch the ZIP64 end-of-central-directory to locate the central
directory, pull just that (~70MB), parse it for the entries whose path
carries an IMDb id we care about, then range-fetch and inflate only those
members. Total transfer lands in the hundreds of MB.

Archive layout is <lang>/<year>/<imdb_id>/<file>.xml.gz, and movies.parquet
already has imdb_id, so the join is exact — no title matching.

Stage 1 (--index) caches the parsed directory; stage 2 (--fetch) downloads.
Both are resumable.

Usage:
    python fetch_subtitles.py --index
    python fetch_subtitles.py --fetch --limit 50    # try a few first
    python fetch_subtitles.py --fetch
"""

import argparse
import io
import json
import re
import struct
import sys
import zlib
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8")

PIPELINE_DIR = Path(__file__).parent
DATA_DIR = PIPELINE_DIR / "data"
SUBS_DIR = DATA_DIR / "subs"
INDEX_PATH = DATA_DIR / "subs_index.json"

URL = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/xml/en.zip"

EOCD64_LOCATOR = b"PK\x06\x07"
CD_ENTRY = b"PK\x01\x02"


def get_range(session, start, length):
    end = start + length - 1
    r = session.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=180)
    r.raise_for_status()
    return r.content


def locate_central_directory(session, total):
    """ZIP64: the tail locator points at the EOCD64 record, which holds the
    central directory's real offset and size (the classic EOCD maxes out at
    4GB and just stores 0xffffffff here)."""
    tail = get_range(session, total - 128, 128)
    i = tail.rfind(EOCD64_LOCATOR)
    if i < 0:
        raise SystemExit("no ZIP64 locator found — archive layout changed?")
    eocd64_off = struct.unpack_from("<Q", tail, i + 8)[0]
    rec = get_range(session, eocd64_off, 56)
    if rec[:4] != b"PK\x06\x06":
        raise SystemExit("ZIP64 EOCD signature missing")
    cd_entries = struct.unpack_from("<Q", rec, 32)[0]
    cd_size = struct.unpack_from("<Q", rec, 40)[0]
    cd_offset = struct.unpack_from("<Q", rec, 48)[0]
    return cd_offset, cd_size, cd_entries


def parse_central_directory(buf, wanted):
    """Walk the central directory, keeping entries for the IMDb ids we want.

    ZIP64 stores oversized values in an extra field rather than inline, so a
    0xffffffff placeholder means the real number lives in extra id 0x0001.
    """
    out = {}
    pos = 0
    n = len(buf)
    pat = re.compile(r"/(\d{4})/(\d+)/[^/]+\.xml\.gz$")
    while pos + 46 <= n:
        if buf[pos:pos + 4] != CD_ENTRY:
            break
        comp_size, uncomp_size = struct.unpack_from("<II", buf, pos + 20)
        name_len, extra_len, cmt_len = struct.unpack_from("<HHH", buf, pos + 28)
        method = struct.unpack_from("<H", buf, pos + 10)[0]
        local_off = struct.unpack_from("<I", buf, pos + 42)[0]
        name = buf[pos + 46: pos + 46 + name_len].decode("utf-8", "replace")
        extra = buf[pos + 46 + name_len: pos + 46 + name_len + extra_len]

        if 0xFFFFFFFF in (comp_size, uncomp_size, local_off):
            e = 0
            while e + 4 <= len(extra):
                hid, hsz = struct.unpack_from("<HH", extra, e)
                if hid == 0x0001:
                    vals = []
                    off = e + 4
                    for orig in (uncomp_size, comp_size, local_off):
                        if orig == 0xFFFFFFFF and off + 8 <= e + 4 + hsz:
                            vals.append(struct.unpack_from("<Q", extra, off)[0])
                            off += 8
                        else:
                            vals.append(orig)
                    uncomp_size, comp_size, local_off = vals
                    break
                e += 4 + hsz

        m = pat.search(name)
        if m:
            imdb = int(m.group(2))
            if imdb in wanted:
                prev = out.get(imdb)
                # Several releases per film; keep the largest, which is the
                # most complete transcription.
                if prev is None or uncomp_size > prev["uncomp"]:
                    out[imdb] = {"name": name, "off": local_off,
                                 "comp": comp_size, "uncomp": uncomp_size,
                                 "method": method}
        pos += 46 + name_len + extra_len + cmt_len
    return out


def fetch_member(session, ent):
    """Range-fetch one member. The local header repeats the name/extra with
    its own lengths, so read it first to find where the data actually starts."""
    head = get_range(session, ent["off"], 30)
    if head[:4] != b"PK\x03\x04":
        return None
    nl, el = struct.unpack_from("<HH", head, 26)
    data = get_range(session, ent["off"] + 30 + nl + el, ent["comp"])
    if ent["method"] == 0:
        raw = data
    else:
        raw = zlib.decompress(data, -zlib.MAX_WBITS)
    # Members are themselves gzipped .xml.gz
    return zlib.decompress(raw, 16 + zlib.MAX_WBITS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", action="store_true", help="build the entry index")
    ap.add_argument("--fetch", action="store_true", help="download members")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    movies = pd.read_parquet(DATA_DIR / "movies.parquet")
    imdb = {}
    for i, v in enumerate(movies["imdb_id"]):
        if isinstance(v, str) and v.startswith("tt"):
            imdb[int(v[2:])] = i
    print(f"{len(imdb)} movies carry an imdb_id")

    session = requests.Session()

    if args.index:
        total = int(session.head(URL, timeout=60).headers["content-length"])
        cd_off, cd_size, cd_n = locate_central_directory(session, total)
        print(f"archive {total/1e9:.1f} GB · central directory "
              f"{cd_size/1e6:.0f} MB, {cd_n} entries — fetching directory only")
        chunks, got = [], 0
        with tqdm(total=cd_size, unit="B", unit_scale=True, desc="index") as bar:
            while got < cd_size:
                take = min(32 << 20, cd_size - got)
                chunks.append(get_range(session, cd_off + got, take))
                got += take
                bar.update(take)
        found = parse_central_directory(b"".join(chunks), set(imdb))
        INDEX_PATH.write_text(json.dumps(
            {str(k): v for k, v in found.items()}))
        print(f"matched {len(found)}/{len(imdb)} movies -> {INDEX_PATH.name}")

    if args.fetch:
        index = {int(k): v for k, v in json.loads(INDEX_PATH.read_text()).items()}
        SUBS_DIR.mkdir(parents=True, exist_ok=True)
        todo = [(k, v) for k, v in index.items()
                if not (SUBS_DIR / f"{imdb[k]}.xml").exists()]
        if args.limit:
            todo = todo[: args.limit]
        print(f"{len(todo)} to fetch ({len(index) - len(todo)} already cached)")
        bytes_got = 0
        for k, ent in tqdm(todo, desc="subs"):
            try:
                xml = fetch_member(session, ent)
            except Exception as e:  # keep going; one bad member isn't fatal
                tqdm.write(f"  skip {k}: {type(e).__name__} {e}")
                continue
            if xml:
                (SUBS_DIR / f"{imdb[k]}.xml").write_bytes(xml)
                bytes_got += ent["comp"]
        print(f"transferred ~{bytes_got/1e6:.0f} MB")


if __name__ == "__main__":
    main()
