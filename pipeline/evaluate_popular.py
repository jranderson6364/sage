"""Check the shipped axes against per-film judgments on the 100 top movies.

Rank correlation on the small held-out set kept passing while user-visible
scores got worse — it can't see compression, and it can't see that a film
with no tone-bearing tags is being scored at the population mean and then
rank-normalized into a confident-looking 58th percentile. This checks the
0-100 values as actually displayed, on films anyone can sanity-check.

Reports signed bias (is the axis systematically high or low?), MAE, and the
worst offenders, so a regression shows up as "Joker's intimacy is 30 points
off" rather than as a third decimal place.

Caveat, deliberately recorded: these labels were written while the model's
own scores were visible, so anchoring is possible. That makes them reliable
for catching large errors — anchoring would pull judgments *toward* the
model, so measured error is if anything understated — but not clean enough
for fine-grained model selection. Use axis_labels.json for that.

Usage:
    python evaluate_popular.py
    python evaluate_popular.py --axes data/axes_experiment.npy
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--axes", default=str(DATA_DIR / "axes_learned.npy"))
    ap.add_argument("--worst", type=int, default=6)
    args = ap.parse_args()

    scores = np.load(args.axes)
    movies = pd.read_parquet(DATA_DIR / "movies.parquet")
    labels = json.loads((PIPELINE_DIR / "axis_labels_popular.json").read_text())["labels"]

    idx = np.array([int(k) for k in labels])
    truth = np.array([labels[k] for k in labels], dtype=float)   # already 0-100

    # Compare against percentile rank, which is what the detail panel shows.
    # Do NOT use (v+1)/2*100 here: the display transform stretches the tails
    # past ±1, so that mapping overflows 0-100 and inflates the error for
    # every extreme film — it reported a phantom MAE regression of ~3 points
    # with the model's ordering completely unchanged.
    pct = np.empty_like(scores, dtype=float)
    for a in range(scores.shape[1]):
        r = np.empty(len(scores))
        r[np.argsort(scores[:, a], kind="stable")] = np.arange(len(scores))
        pct[:, a] = (r + 0.5) / len(scores) * 100
    pred = pct[idx]
    err = pred - truth

    print(f"{len(idx)} popular films · axes: {Path(args.axes).name}\n")
    print(f"{'axis':10} {'bias':>7} {'MAE':>7} {'>20 off':>9}")
    for a, name in enumerate(AXIS_NAMES):
        e = err[:, a]
        print(f"{name:10} {e.mean():+7.1f} {np.abs(e).mean():7.1f} "
              f"{int((np.abs(e) > 20).sum()):6d}/100")

    for a, name in enumerate(AXIS_NAMES):
        order = np.argsort(-np.abs(err[:, a]))[: args.worst]
        print(f"\nworst {name}:")
        for j in order:
            t = movies["title"].iloc[idx[j]]
            print(f"  {t[:42]:42} want {truth[j, a]:3.0f}  got {pred[j, a]:3.0f}  "
                  f"({err[j, a]:+.0f})")

    print("\nbias sign: + means the model scores this axis too high overall")


if __name__ == "__main__":
    main()
