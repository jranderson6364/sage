"""Score axis values against the pairwise comparison ground truth.

This is the strictest check available, and the most directly meaningful:
"what fraction of observed comparisons does the model get the right way
round". Unlike correlation against 1-10 labels it needs no shared scale, and
unlike the popular-100 spot check it can't be contaminated by having seen
the model's output first — the judgments were orderings, made without any
scores in view.

Also reports accuracy split by how far apart the two films are on the fitted
Bradley-Terry scale. Near-ties are genuinely ambiguous and getting them wrong
costs little; missing a wide-gap pair (calling Paddington scarier than The
Exorcist) is a real failure, so they're worth reading separately.

Usage:
    python evaluate_bt.py
    python evaluate_bt.py --axes data/axes_experiment.npy
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from fit_bradley_terry import pairs_from_rankings

sys.stdout.reconfigure(encoding="utf-8")

PIPELINE_DIR = Path(__file__).parent
DATA_DIR = PIPELINE_DIR / "data"
AXIS_NAMES = ["levity", "threat", "intimacy"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--axes", default=str(DATA_DIR / "axes_learned.npy"))
    args = ap.parse_args()

    scores = np.load(args.axes)
    movies = pd.read_parquet(DATA_DIR / "movies.parquet")
    print(f"axes: {Path(args.axes).name}\n")
    print(f"{'axis':10} {'pairs':>7} {'overall':>9} {'wide gap':>10} "
          f"{'near tie':>10} {'rho vs BT':>11}")

    for axis in AXIS_NAMES:
        pf = DATA_DIR / f"pairs_{axis}.json"
        rf = PIPELINE_DIR / f"rankings_{axis}.json"
        bf = DATA_DIR / f"bt_{axis}.json"
        if not (pf.exists() and rf.exists() and bf.exists()):
            print(f"{axis:10}   (no comparisons yet)")
            continue

        spec = json.loads(pf.read_text())
        ranks = json.loads(rf.read_text())["rankings"]
        bt = json.loads(bf.read_text())
        pairs = pairs_from_rankings(spec["groups"], ranks)
        col = AXIS_NAMES.index(axis)

        s = {f: v for f, v in zip(bt["films"], bt["scores"])}
        gaps = np.array([abs(s[a] - s[b]) for a, b in pairs])
        correct = np.array([scores[a, col] > scores[b, col] for a, b in pairs])

        wide = gaps >= np.percentile(gaps, 66)
        near = gaps <= np.percentile(gaps, 33)
        rho = spearmanr([s[f] for f in bt["films"]],
                        scores[bt["films"], col]).statistic
        print(f"{axis:10} {len(pairs):7d} {correct.mean():8.1%} "
              f"{correct[wide].mean():9.1%} {correct[near].mean():9.1%} "
              f"{rho:11.3f}")

    print("\nwide gap = clearly different films; getting these wrong is a real error.")
    print("near tie = genuinely ambiguous pairs; ~50-70% is expected there.")


if __name__ == "__main__":
    main()
