"""Draw comparison groups for pairwise (Bradley-Terry) axis scoring.

Absolute 1-10 ratings drift: the same judge calls a film 7 today and 5 next
week, because "7" has no fixed referent. Comparisons don't drift — "is A
scarier than B" is stable — which is why psychometrics prefers them.

Rather than emit single pairs, this emits *groups* to rank. One ranking of 8
films yields 28 pairwise comparisons and is far more internally consistent
than 28 independent judgments (a judge ranking a list can't contradict
itself the way separate answers can).

Two sampling requirements:
  connectivity  Bradley-Terry is only identifiable if the comparison graph
                is connected, so every film appears in several groups.
  anchors       Pole-defining films are force-included and salted across
                groups, so the fitted scale is pinned to known extremes
                instead of floating free.

Films are stratified by the current model's score so groups span the range
rather than clustering, and drawn from the most-voted titles so the judge
actually knows them.

Usage:
    python sample_pairs.py --axis threat --films 64 --per-film 3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

PIPELINE_DIR = Path(__file__).parent
DATA_DIR = PIPELINE_DIR / "data"
AXIS_NAMES = ["levity", "threat", "intimacy"]
SEED = 20260722
GROUP = 8

# Pole-defining films. These fix what the ends of each scale *mean*, so the
# fitted scale is anchored to "as scary as films get" rather than to "top of
# whatever happens to be in this catalog".
ANCHORS = {
    "threat": {
        "high": ["The Exorcist", "Hereditary", "The Shining", "Sinister", "Terrifier"],
        "low": ["Paddington", "Ratatouille", "My Neighbor Totoro", "Legally Blonde",
                "Love Actually"],
    },
    "levity": {
        "high": ["Airplane!", "The Naked Gun: From the Files of Police Squad!",
                 "Monty Python and the Holy Grail", "Superbad", "Anchorman: The Legend of Ron Burgundy"],
        "low": ["Schindler's List", "Come and See", "Grave of the Fireflies",
                "Requiem for a Dream", "The Road"],
    },
    "intimacy": {
        "high": ["The Notebook", "Before Sunrise", "Call Me by Your Name",
                 "Brokeback Mountain", "Portrait of a Lady on Fire"],
        "low": ["Mad Max: Fury Road", "Dunkirk", "No Country for Old Men",
                "The Raid", "Koyaanisqatsi"],
    },
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--axis", default="threat", choices=AXIS_NAMES)
    ap.add_argument("--films", type=int, default=64)
    ap.add_argument("--per-film", type=int, default=3, help="groups each film joins")
    ap.add_argument("--min-votes", type=int, default=2500)
    args = ap.parse_args()

    movies = pd.read_parquet(DATA_DIR / "movies.parquet")
    axes = np.load(DATA_DIR / "axes_learned.npy")
    col = AXIS_NAMES.index(args.axis)
    rng = np.random.default_rng(SEED)

    anchors = ANCHORS[args.axis]
    forced = []
    for t in anchors["high"] + anchors["low"]:
        hit = movies.index[movies["title"] == t]
        if len(hit):
            forced.append(int(hit[0]))
        else:
            print(f"# anchor not in catalog, skipped: {t}")
    forced = list(dict.fromkeys(forced))

    # Stratify the rest across the current score so groups span the range.
    pool = movies.index[movies["vote_count"] >= args.min_votes]
    pool = np.array([i for i in pool if i not in set(forced)])
    order = pool[np.argsort(axes[pool, col])]
    need = max(args.films - len(forced), 0)
    picks = [order[int(round(q))] for q in
             np.linspace(0, len(order) - 1, need)] if need else []
    films = list(dict.fromkeys(forced + [int(i) for i in picks]))

    # Build groups: repeat the film list `per_film` times, shuffle each pass,
    # and chunk. Every film lands in that many groups, and the overlap between
    # passes is what connects the comparison graph.
    groups = []
    for _ in range(args.per_film):
        shuffled = films.copy()
        rng.shuffle(shuffled)
        for s in range(0, len(shuffled) - GROUP + 1, GROUP):
            groups.append(shuffled[s:s + GROUP])

    out = {"axis": args.axis, "films": films, "groups": groups,
           "anchors": {k: [int(movies.index[movies["title"] == t][0])
                           for t in v if len(movies.index[movies["title"] == t])]
                       for k, v in anchors.items()}}
    path = DATA_DIR / f"pairs_{args.axis}.json"
    path.write_text(json.dumps(out), encoding="utf-8")
    print(f"# {len(films)} films, {len(groups)} groups of {GROUP} "
          f"= {len(groups) * GROUP * (GROUP - 1) // 2} pairwise comparisons")
    print(f"# wrote {path.name}\n")
    for gi, g in enumerate(groups):
        titles = " | ".join(f"{movies['title'].iloc[i]} ({movies['year'].iloc[i]})"
                            for i in g)
        print(f"G{gi:02d}\t{titles}")


if __name__ == "__main__":
    main()
