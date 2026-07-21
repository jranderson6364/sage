"""Score the semantic axes against a hand-labeled validation set.

Until this existed there was no metric for axis quality — only a handful of
eyeballed calibration titles — so any change to the scoring was unfalsifiable.

Reports three things per axis, because a single correlation hides the failure
mode that matters most here:

  rho       Spearman correlation with the labels. The headline number.
  covered / uncovered
            The same rho split by genome coverage. Uncovered movies have no
            community tags to anchor them and are where the scorer is weakest,
            so an aggregate that lumps them in with 80% covered movies can
            improve while that group gets worse.
  sep       Mean model percentile of the movies labeled high (>=8) minus
            that of the movies labeled low (<=3). This is the guard against
            "fixes" that raise rho by regressing everything toward the mean:
            flattened extremes still correlate fine (order is preserved)
            while making the 3D cloud a featureless blob. Deliberately a
            difference of two means rather than an error against the labels,
            since a 1-10 label doesn't map linearly onto a percentile — but
            the collapse it's watching for shows up the same way regardless
            of that mapping, and it's comparable across runs.

Usage:
    python evaluate_axes.py
    python evaluate_axes.py --axes data/axes_experiment.npy   # compare a variant
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PIPELINE_DIR = Path(__file__).parent
DATA_DIR = PIPELINE_DIR / "data"

AXIS_NAMES = ["levity", "threat", "intimacy"]
LOW, HIGH = 3, 8  # label values counted as "extreme" for the tail check


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axes", default=str(DATA_DIR / "axes.npy"))
    args = parser.parse_args()

    scores = np.load(args.axes)
    movies = pd.read_parquet(DATA_DIR / "movies.parquet")
    labels = json.loads((PIPELINE_DIR / "axis_labels.json").read_text())["labels"]
    genome_rows = set(json.loads((DATA_DIR / "genome_rows.json").read_text()))

    idx = np.array([int(k) for k in labels])
    truth = np.array([labels[k] for k in labels], dtype=float)  # n x 3, 1..10
    is_cov = np.array([i in genome_rows for i in idx])
    # Convert to within-catalog percentile before measuring spread. Some
    # scorers emit forced-uniform percentiles and some emit a predicted 1-10
    # rating, and raw spreads across those two parameterizations aren't
    # comparable — ranking first makes `sep` mean the same thing either way.
    ranked = np.empty_like(scores, dtype=float)
    for a in range(scores.shape[1]):
        r = np.empty(len(scores))
        r[np.argsort(scores[:, a], kind="stable")] = np.arange(len(scores))
        ranked[:, a] = (r + 0.5) / len(scores) * 100
    pred = ranked[idx]

    print(f"{len(idx)} labeled movies "
          f"({is_cov.sum()} genome-covered, {(~is_cov).sum()} uncovered)")
    print(f"axes: {args.axes}\n")
    print(f"{'axis':10} {'rho':>6} {'covered':>9} {'uncovered':>10} "
          f"{'low':>6} {'high':>6} {'sep':>7}")

    for a, name in enumerate(AXIS_NAMES):
        t, p = truth[:, a], pred[:, a]
        rho = spearmanr(t, p).statistic
        rho_cov = spearmanr(t[is_cov], p[is_cov]).statistic
        rho_unc = spearmanr(t[~is_cov], p[~is_cov]).statistic

        lo, hi = t <= LOW, t >= HIGH
        lo_mean = float(p[lo].mean()) if lo.any() else float("nan")
        hi_mean = float(p[hi].mean()) if hi.any() else float("nan")

        print(f"{name:10} {rho:6.3f} {rho_cov:9.3f} {rho_unc:10.3f} "
              f"{lo_mean:6.1f} {hi_mean:6.1f} {hi_mean - lo_mean:7.1f}")

    print(f"\nlow/high = mean model percentile of movies labeled <={LOW} / >={HIGH}")
    print("sep      = the gap between them. Watch this across runs: a change that")
    print("           lifts rho while shrinking sep is flattening the cloud, not")
    print("           improving it.")


if __name__ == "__main__":
    main()
