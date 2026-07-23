"""Resolve the list definitions in lists.json against our catalogue.

Two kinds of definition (see lists.json):
  tmdb    - a public TMDB list, pulled by id. Joins on tmdb_id, so the match is
            exact; cached one JSON per list so re-runs are free.
  curated - hand-picked [title, year] pairs, matched against movies.parquet.
            Title+year keeps the definition auditable; anything that fails to
            resolve is printed rather than silently dropped.

The interesting number here is *coverage*. Our catalogue is TMDB's top ~5000 by
vote count, so a list of popular films lands near 100% while an arthouse-leaning
one barely registers (measured: Golden Lion 11%, Palme d'Or 30%). A list that
matches a third of its films isn't that list any more, so --min-coverage drops
those instead of shipping a misleading stub.

Writes pipeline/data/lists_resolved.json: slug -> {name, description, rows, ...}
where `rows` are row indices into movies.parquet.

Usage:
    python fetch_lists.py
    python fetch_lists.py --min-coverage 0.5    # default 0.65
    python fetch_lists.py --refresh             # ignore cached TMDB pulls
"""

import argparse
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

PIPELINE_DIR = Path(__file__).parent
REPO_ROOT = PIPELINE_DIR.parent
DATA_DIR = PIPELINE_DIR / "data"
RAW_DIR = DATA_DIR / "raw_lists"

API_BASE = "https://api.themoviedb.org/3"
REQUEST_DELAY_S = 0.15

# A list needs to keep this fraction of its films to be worth shipping.
MIN_COVERAGE = 0.65


def make_session() -> requests.Session:
    load_dotenv(REPO_ROOT / ".env")
    token = os.getenv("TMDB_READ_ACCESS_TOKEN")
    session = requests.Session()
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    else:
        key = os.getenv("TMDB_API_KEY")
        if not key:
            raise SystemExit(
                "No TMDB credentials found. Put TMDB_READ_ACCESS_TOKEN (preferred) "
                f"or TMDB_API_KEY in {REPO_ROOT / '.env'}"
            )
        session.params = {"api_key": key}
    return session


def get(session: requests.Session, path: str, **params) -> dict:
    for _ in range(5):
        resp = session.get(f"{API_BASE}{path}", params=params, timeout=30)
        if resp.status_code == 429:
            time.sleep(int(resp.headers.get("Retry-After", 2)) + 1)
            continue
        resp.raise_for_status()
        time.sleep(REQUEST_DELAY_S)
        return resp.json()
    raise RuntimeError(f"Giving up on {path} after repeated 429s")


def fetch_tmdb_list(session: requests.Session, list_id: int, refresh: bool) -> list[int]:
    """Every movie tmdb_id on a public TMDB list, following pagination."""
    cache = RAW_DIR / f"{list_id}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))

    ids: list[int] = []
    page = 1
    while True:
        data = get(session, f"/list/{list_id}", page=page)
        items = data.get("items", [])
        # Lists can hold TV as well as film; we only map movies.
        ids += [i["id"] for i in items if i.get("media_type", "movie") == "movie"]
        if not items or page >= data.get("total_pages", 1):
            break
        page += 1

    ids = list(dict.fromkeys(ids))
    cache.write_text(json.dumps(ids), encoding="utf-8")
    return ids


def norm_title(s: str) -> str:
    """Fold a title to something matchable: accents, punctuation, articles.

    'Amélie' / 'Amelie' and '(500) Days of Summer' / '500 Days of Summer' should
    match; the year is what actually disambiguates, so this can be aggressive.
    """
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


def resolve_curated(films: list, by_title: dict) -> tuple[list[int], list]:
    """Map [title, year] pairs to tmdb_ids, tolerating a +/-1 year drift.

    Release year differs between TMDB and common usage for festival-run and
    late-December films, so an exact-year-only match drops real hits.
    """
    found, missing = [], []
    for title, year in films:
        key = norm_title(title)
        hit = by_title.get((key, year))
        if hit is None:
            for delta in (-1, 1):
                hit = by_title.get((key, year + delta))
                if hit is not None:
                    break
        if hit is None:
            missing.append([title, year])
        else:
            found.append(hit)
    return list(dict.fromkeys(found)), missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-coverage", type=float, default=MIN_COVERAGE,
                        help="drop lists matching less than this fraction")
    parser.add_argument("--refresh", action="store_true",
                        help="re-pull TMDB lists instead of using the cache")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    defs = json.loads((PIPELINE_DIR / "lists.json").read_text(encoding="utf-8"))

    movies = pd.read_parquet(DATA_DIR / "movies.parquet")
    row_of_tmdb = {int(t): i for i, t in enumerate(movies["tmdb_id"])}
    by_title: dict[tuple[str, int], int] = {}
    for i, row in enumerate(movies.itertuples()):
        if row.year:
            # First writer wins: movies.parquet is ordered by vote count, so a
            # title collision resolves to the better-known film.
            by_title.setdefault((norm_title(row.title), int(row.year)), int(row.tmdb_id))

    session = None
    resolved: dict[str, dict] = {}
    dropped: list[tuple[str, int, int]] = []

    for spec in defs["lists"]:
        slug, kind = spec["slug"], spec["kind"]
        if kind == "tmdb":
            if session is None:
                session = make_session()
            tmdb_ids = fetch_tmdb_list(session, spec["tmdb_list_id"], args.refresh)
            missing = []
        elif kind == "curated":
            tmdb_ids, missing = resolve_curated(spec["films"], by_title)
            # A curated title that resolves to nothing is a typo in lists.json,
            # not a catalogue gap — surface it loudly.
            for title, year in missing:
                print(f"  ! {slug}: no catalogue match for {title!r} ({year})")
        else:
            raise SystemExit(f"{slug}: unknown kind {kind!r}")

        rows = [row_of_tmdb[t] for t in tmdb_ids if t in row_of_tmdb]
        total = len(tmdb_ids) + len(missing)
        coverage = len(rows) / max(total, 1)

        if coverage < args.min_coverage or len(rows) < 2:
            dropped.append((spec["name"], len(rows), total))
            continue

        resolved[slug] = {
            "name": spec["name"],
            "description": spec.get("description", ""),
            "source": "TMDB list" if kind == "tmdb" else "curated",
            "rows": rows,
            "matched": len(rows),
            "total": total,
        }
        print(f"{spec['name']:<28} {len(rows):>3}/{total:<4} {coverage:5.0%}")

    for name, matched, total in dropped:
        print(f"DROPPED {name!r}: only {matched}/{total} films in the catalogue")

    out = DATA_DIR / "lists_resolved.json"
    out.write_text(json.dumps(resolved), encoding="utf-8")
    print(f"\nWrote {len(resolved)} lists -> {out}")


if __name__ == "__main__":
    main()
