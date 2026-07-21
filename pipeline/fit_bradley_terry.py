"""Fit a Bradley-Terry latent scale from ranked comparison groups.

Bradley-Terry models P(A beats B) = sigmoid(s_A - s_B) and recovers the
latent strengths s from observed outcomes. Applied here, "beats" means "is
more tense than", so s is a threat scale learned purely from orderings —
no one ever had to say what a "7" means.

Why this beats absolute ratings:
  - Comparisons don't drift. Absolute ratings do: the same judge calls a film
    7 today and 5 next week because "7" has no referent.
  - The spacing is *earned*. Two films that always trade places end up close
    together; a film that beats everything ends up far out. That's real
    separation, not a display transform pushing points around.
  - It's unbounded, so genuine outliers can sit far from the pack without
    anyone choosing how far.

Fitting: maximum likelihood with a small L2 pull toward 0 (which both breaks
the additive degeneracy — only differences are identified — and keeps films
with near-perfect records from running off to infinity).

Reads data/pairs_<axis>.json + rankings_<axis>.json, writes
data/bt_<axis>.json with a fitted score per film.

Usage:
    python fit_bradley_terry.py --axis threat
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8")

PIPELINE_DIR = Path(__file__).parent
DATA_DIR = PIPELINE_DIR / "data"
AXIS_NAMES = ["levity", "threat", "intimacy"]


def pairs_from_rankings(groups, rankings):
    """Expand each ranking into its pairwise comparisons.

    A ranking [a, b, c] means a beat b, a beat c, b beat c — every ordered
    pair, which is what makes one 8-film ranking worth 28 comparisons.
    """
    out = []
    for gi, order in rankings.items():
        g = groups[int(gi)]
        ranked = [g[p] for p in order]
        for i in range(len(ranked)):
            for j in range(i + 1, len(ranked)):
                out.append((ranked[i], ranked[j]))  # winner, loser
    return out


def fit(pairs, films, reg=0.01):
    idx = {f: i for i, f in enumerate(films)}
    w = np.array([idx[a] for a, _ in pairs])
    l = np.array([idx[b] for _, b in pairs])
    n = len(films)

    def nll(s):
        d = s[w] - s[l]
        # log(1 + exp(-d)), stable form
        loss = np.logaddexp(0.0, -d).sum() + reg * np.dot(s, s)
        return loss

    def grad(s):
        d = s[w] - s[l]
        p = 1.0 / (1.0 + np.exp(d))    # dloss/dd
        g = np.zeros(n)
        np.add.at(g, w, -p)
        np.add.at(g, l, p)
        return g + 2 * reg * s

    res = minimize(nll, np.zeros(n), jac=grad, method="L-BFGS-B")
    s = res.x - res.x.mean()
    return s, res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--axis", default="threat", choices=AXIS_NAMES)
    args = ap.parse_args()

    spec = json.loads((DATA_DIR / f"pairs_{args.axis}.json").read_text())
    ranks = json.loads((PIPELINE_DIR / f"rankings_{args.axis}.json").read_text())["rankings"]
    movies = pd.read_parquet(DATA_DIR / "movies.parquet")
    axes = np.load(DATA_DIR / "axes_learned.npy")
    col = AXIS_NAMES.index(args.axis)

    films = spec["films"]
    pairs = pairs_from_rankings(spec["groups"], ranks)
    print(f"{len(films)} films · {len(ranks)} rankings · {len(pairs)} pairwise comparisons")

    # Connectivity: BT is only identifiable on a connected comparison graph.
    seen = {f: set() for f in films}
    for a, b in pairs:
        seen[a].add(b)
        seen[b].add(a)
    deg = np.array([len(seen[f]) for f in films])
    print(f"each film compared against {deg.min()}-{deg.max()} others "
          f"(median {int(np.median(deg))})")

    s, res = fit(pairs, films)
    print(f"converged={res.success} · scale spread {s.min():+.2f}..{s.max():+.2f}")

    # How often does the fitted scale agree with the raw comparisons? A low
    # number would mean the judgments are self-contradictory.
    idx = {f: i for i, f in enumerate(films)}
    agree = np.mean([s[idx[a]] > s[idx[b]] for a, b in pairs])
    print(f"fitted scale reproduces {agree:.1%} of the observed comparisons")

    cur = axes[films, col]
    rho = spearmanr(s, cur).statistic
    print(f"agreement with the current shipped model: rho {rho:.3f}\n")

    order = np.argsort(-s)
    print("most tense (Bradley-Terry):")
    for k in order[:10]:
        i = films[k]
        print(f"  {s[k]:+6.2f}  {movies['title'].iloc[i][:44]:44} "
              f"(model pct {round((cur[k] + 1) / 2 * 100):3d})")
    print("\nleast tense:")
    for k in order[-8:]:
        i = films[k]
        print(f"  {s[k]:+6.2f}  {movies['title'].iloc[i][:44]:44} "
              f"(model pct {round((cur[k] + 1) / 2 * 100):3d})")

    print("\nbiggest disagreements with the current model:")
    bt_rank = np.empty(len(films)); bt_rank[np.argsort(s)] = np.arange(len(films))
    cur_rank = np.empty(len(films)); cur_rank[np.argsort(cur)] = np.arange(len(films))
    gap = bt_rank - cur_rank
    for k in np.argsort(-np.abs(gap))[:8]:
        i = films[k]
        d = "model too LOW" if gap[k] > 0 else "model too HIGH"
        print(f"  {movies['title'].iloc[i][:40]:40} {d} by {abs(gap[k]):.0f} places")

    out = {"axis": args.axis, "films": films,
           "scores": [round(float(v), 4) for v in s],
           "agreement": round(float(agree), 4)}
    (DATA_DIR / f"bt_{args.axis}.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"\nwrote data/bt_{args.axis}.json")


if __name__ == "__main__":
    main()
