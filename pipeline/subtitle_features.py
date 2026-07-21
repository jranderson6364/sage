"""Turn subtitle timing into features the other channels can't see.

Tag genomes and plot summaries describe what a film is *about*. Subtitles
describe how it actually *plays* — and the timing matters as much as the
words:

  silence      Long dialogue-free stretches. Horror and thrillers sit in
               silence; comedies almost never do. This is the feature no
               other channel has any access to.
  density      Words per minute of runtime, and its variance.
  bursts       Rapid-fire exchanges (short gaps) — banter, arguments.
  exclaim/ask  Punctuation rates, a cheap proxy for shouting and interrogation.
  pronouns     "I"/"you" density, which rises in two-hander intimate scenes
               and falls in plot-driven ensemble action.
  lexicon      Small hand-built cue lists (laughter, profanity, terms of
               endearment, violence words).

Every feature is also computed per decile of runtime, giving a shape rather
than a scalar — that's what narrative_arcs.py clusters.

Reads data/subs/*.xml (fetch_subtitles.py), writes data/sub_features.npy,
data/sub_feature_names.json, data/sub_rows.json, and data/sub_arcs.npy
(n x 3 x 10 per-decile curves).

Usage:
    python subtitle_features.py
"""

import json
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent / "data"
SUBS_DIR = DATA_DIR / "subs"
DECILES = 10

CUES = {
    "laugh": {"haha", "hah", "laughs", "laughing", "laughter", "chuckles", "giggles"},
    "scream": {"screams", "screaming", "aaah", "aah", "argh", "gasps", "shrieks"},
    "profan": {"fuck", "fucking", "shit", "damn", "hell", "bastard", "bitch", "ass"},
    "love": {"love", "darling", "honey", "sweetheart", "baby", "dear", "kiss", "marry"},
    "violence": {"kill", "killed", "die", "dead", "blood", "gun", "shoot", "run",
                 "help", "stop", "no"},
    "family": {"mom", "mother", "dad", "father", "son", "daughter", "brother",
               "sister", "family", "grandma", "grandpa"},
}
PRONOUN = {"i", "you", "me", "my", "your", "we", "us"}

TIME_RE = re.compile(r"(\d+):(\d+):(\d+),(\d+)")


def to_seconds(v):
    m = TIME_RE.match(v or "")
    if not m:
        return None
    h, mi, s, ms = (int(x) for x in m.groups())
    return h * 3600 + mi * 60 + s + ms / 1000


def parse(path):
    """Return (times, words) for one subtitle file, or None if unusable."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None
    events = []
    for s in root.iter("s"):
        t = None
        words = []
        for el in s:
            if el.tag == "time":
                sec = to_seconds(el.get("value"))
                if sec is not None and t is None:
                    t = sec
            elif el.tag == "w" and el.text:
                words.append(el.text)
        if words:
            events.append((t, words))
    if len(events) < 100:
        return None

    # Timestamps only appear on some <s> nodes; carry the last known time
    # forward so every line lands somewhere on the runtime.
    last = 0.0
    out = []
    for t, w in events:
        if t is not None:
            last = t
        out.append((last, w))
    if out[-1][0] < 600:  # under 10 minutes of timeline — broken or a clip
        return None
    return out


def features(events):
    total = events[-1][0]
    times = np.array([t for t, _ in events])
    counts = np.array([len(w) for _, w in events])
    gaps = np.diff(times)
    gaps = gaps[(gaps >= 0) & (gaps < 600)]

    allw = [w.lower() for _, ws in events for w in ws]
    nw = max(len(allw), 1)

    def rate(sett):
        return sum(1 for w in allw if w in sett) / nw

    flat = {
        "runtime_min": total / 60,
        "wpm": nw / max(total / 60, 1),
        "line_len_mean": counts.mean(),
        "gap_mean": gaps.mean() if len(gaps) else 0,
        "gap_p90": np.percentile(gaps, 90) if len(gaps) else 0,
        # The tension signal: share of the film with no dialogue for 10s / 30s.
        "silence_10s": float((gaps > 10).sum() * 1.0 / max(len(gaps), 1)),
        "silence_30s": float((gaps > 30).sum() * 1.0 / max(len(gaps), 1)),
        "longest_silence": float(gaps.max()) if len(gaps) else 0,
        "burst_rate": float((gaps < 1.0).sum() / max(len(gaps), 1)),
        "exclaim": sum(w.count("!") for w in allw) / nw,
        "question": sum(w.count("?") for w in allw) / nw,
        "pronoun": rate(PRONOUN),
    }
    for k, s in CUES.items():
        flat[f"cue_{k}"] = rate(s)

    # Per-decile curves for three quantities that plausibly trace an arc.
    edges = np.linspace(0, total, DECILES + 1)
    arcs = np.zeros((3, DECILES), dtype=np.float32)
    for d in range(DECILES):
        m = (times >= edges[d]) & (times < edges[d + 1])
        if not m.any():
            continue
        seg_words = [w.lower() for (t, ws) in events
                     if edges[d] <= t < edges[d + 1] for w in ws]
        sn = max(len(seg_words), 1)
        seg_gaps = np.diff(times[m])
        arcs[0, d] = sn / max((edges[d + 1] - edges[d]) / 60, 1e-6)  # density
        arcs[1, d] = (seg_gaps > 10).mean() if len(seg_gaps) else 0   # silence
        arcs[2, d] = sum(1 for w in seg_words
                         if w in CUES["violence"] or w in CUES["scream"]) / sn
    return flat, arcs


def main() -> None:
    files = sorted(SUBS_DIR.glob("*.xml"), key=lambda p: int(p.stem))
    print(f"{len(files)} subtitle files")
    rows, feats, arcs, names = [], [], [], None
    skipped = 0
    for p in tqdm(files, desc="parse"):
        ev = parse(p)
        if ev is None:
            skipped += 1
            continue
        flat, arc = features(ev)
        if names is None:
            names = list(flat)
        rows.append(int(p.stem))
        feats.append([flat[k] for k in names])
        arcs.append(arc)

    F = np.array(feats, dtype=np.float32)
    A = np.stack(arcs).astype(np.float32)
    np.save(DATA_DIR / "sub_features.npy", F)
    np.save(DATA_DIR / "sub_arcs.npy", A)
    (DATA_DIR / "sub_rows.json").write_text(json.dumps(rows))
    (DATA_DIR / "sub_feature_names.json").write_text(json.dumps(names))
    print(f"usable {len(rows)} · skipped {skipped} (too short / unparseable)")
    print(f"features {F.shape} · arcs {A.shape}")
    for i, k in enumerate(names):
        print(f"  {k:16} mean {F[:, i].mean():8.3f}  sd {F[:, i].std():8.3f}")


if __name__ == "__main__":
    main()
