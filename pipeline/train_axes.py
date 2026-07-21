"""Learn the semantic axes from labels instead of hand-tuning them.

The hand-built scorer (semantic_axes.py) sums ~24 hand-picked tag dimensions
out of 1,128 per axis, then blends three channels with weights tuned by hand.
This replaces both steps with a fitted model over the full feature space, so
feature selection and channel weighting are learned rather than guessed.

Features per movie (every one dense — no missingness, thanks to kNN
imputation of both sparse sources):
  genome      1,128 tag relevances (imputed for the ~991 uncovered movies)
  text        384-d story embedding
  review      384-d mean-pooled review embedding (imputed where absent)

Models compared, selected by cross-validation on the training labels:
  ridge_genome   ridge on tags only — the strongest single source
  ridge_all      ridge on everything
  gbm            gradient boosting on everything (nonlinear interactions)

Trained on axis_labels_train.json, reported on axis_labels.json, which is
held out and never fitted. Note the test set was used to tune the *hand*
weights, so it slightly favors the baseline — any win here is conservative.

A regression onto the 1-10 label scale is also the calibration fix: the
output is a predicted human rating, not a forced-uniform percentile, so the
axis distributions come out shaped like the real ones instead of flat.

Writes data/axes_learned.npy in the same [-1, 1] layout as semantic_axes.py.

Usage:
    python train_axes.py
    python train_axes.py --model ridge_all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")

PIPELINE_DIR = Path(__file__).parent
DATA_DIR = PIPELINE_DIR / "data"
AXIS_NAMES = ["levity", "threat", "intimacy"]
LOW, HIGH = 3, 8
SEED = 20260721


def knn_impute(emb, values, have_rows, n, k=10, power=5.0):
    """Fill rows missing `values` from their nearest neighbors that have it.

    Same approach as semantic_axes.impute_genome, applied to any block of
    features; sharpened weights keep borrowed rows from collapsing to the
    dataset mean.
    """
    have = np.asarray(have_rows)
    mask = np.zeros(n, dtype=bool)
    mask[have] = True
    missing = np.flatnonzero(~mask)

    out = np.zeros((n, values.shape[1]), dtype=np.float32)
    out[have] = values
    if len(missing) == 0:
        return out
    sim = emb[missing] @ emb[have].T
    top = np.argpartition(-sim, k, axis=1)[:, :k]
    rows = np.arange(len(missing))[:, None]
    w = np.clip(sim[rows, top], 0, None) ** power
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    out[missing] = np.einsum("ik,ikd->id", w, values[top])
    return out


def report(name, truth, pred, is_cov):
    """rho / MAE / extremes separation, matching evaluate_axes.py's columns."""
    rho = spearmanr(truth, pred).statistic
    rho_c = spearmanr(truth[is_cov], pred[is_cov]).statistic if is_cov.any() else np.nan
    rho_u = spearmanr(truth[~is_cov], pred[~is_cov]).statistic if (~is_cov).any() else np.nan
    mae = float(np.abs(truth - pred).mean())
    lo, hi = truth <= LOW, truth >= HIGH
    sep = float(pred[hi].mean() - pred[lo].mean())
    print(f"  {name:14} rho {rho:.3f}  cov {rho_c:.3f}  unc {rho_u:.3f}  "
          f"MAE {mae:.2f}  sep {sep:.2f}")
    return rho


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None,
                        help="force a model instead of picking by CV")
    parser.add_argument("--out", default=str(DATA_DIR / "axes_learned.npy"))
    args = parser.parse_args()

    movies = pd.read_parquet(DATA_DIR / "movies.parquet")
    n = len(movies)
    emb = np.load(DATA_DIR / "text_emb.npy")
    genome = np.load(DATA_DIR / "genome.npy")
    genome_rows = json.loads((DATA_DIR / "genome_rows.json").read_text())
    tag_names = json.loads((DATA_DIR / "genome_tags.json").read_text())
    review_emb = np.load(DATA_DIR / "review_emb.npy")
    review_rows = json.loads((DATA_DIR / "review_rows.json").read_text())

    genome_all = knn_impute(emb, genome, genome_rows, n)
    review_all = knn_impute(emb, review_emb, review_rows, n)
    X = {
        "ridge_genome": genome_all,
        "ridge_all": np.hstack([genome_all, emb, review_all]),
        "gbm": np.hstack([genome_all, emb, review_all]),
    }

    train = json.loads((PIPELINE_DIR / "axis_labels_train.json").read_text())["labels"]
    test = json.loads((PIPELINE_DIR / "axis_labels.json").read_text())["labels"]
    tr_idx = np.array([int(k) for k in train])
    te_idx = np.array([int(k) for k in test])
    y_tr = np.array([train[k] for k in train], dtype=float)
    y_te = np.array([test[k] for k in test], dtype=float)
    cov = set(genome_rows)
    te_cov = np.array([i in cov for i in te_idx])
    print(f"train {len(tr_idx)} · test {len(te_idx)} "
          f"({te_cov.sum()} genome-covered, {(~te_cov).sum()} uncovered)\n")

    def build(kind, alpha=None):
        if kind == "gbm":
            return HistGradientBoostingRegressor(
                max_depth=3, max_iter=300, learning_rate=0.06, random_state=SEED)
        return Ridge(alpha=alpha)

    preds_all = {}
    chosen = {}
    for a, axis in enumerate(AXIS_NAMES):
        print(f"[{axis}]")
        best = (None, -np.inf, None)
        for kind in (["ridge_genome", "ridge_all", "gbm"] if not args.model
                     else [args.model]):
            feats = X[kind]
            sc = StandardScaler().fit(feats[tr_idx])
            Xtr = sc.transform(feats[tr_idx])
            cv = KFold(5, shuffle=True, random_state=SEED)
            if kind.startswith("ridge"):
                # Pick regularization by CV — with ~1.9k features and 279
                # rows this is what keeps it from memorizing.
                best_a, best_s = None, -np.inf
                for alpha in [1, 10, 30, 100, 300, 1000, 3000, 10000]:
                    p = cross_val_predict(build(kind, alpha), Xtr, y_tr[:, a], cv=cv)
                    s = spearmanr(y_tr[:, a], p).statistic
                    if s > best_s:
                        best_a, best_s = alpha, s
                model, cv_rho, tag = build(kind, best_a), best_s, f"{kind}(a={best_a})"
            else:
                p = cross_val_predict(build(kind), Xtr, y_tr[:, a], cv=cv)
                model, cv_rho, tag = build(kind), spearmanr(y_tr[:, a], p).statistic, kind
            print(f"  {tag:22} cv_rho {cv_rho:.3f}")
            if cv_rho > best[1]:
                best = (kind, cv_rho, model)

        kind, cv_rho, model = best
        feats = X[kind]
        sc = StandardScaler().fit(feats[tr_idx])
        model.fit(sc.transform(feats[tr_idx]), y_tr[:, a])
        full = model.predict(sc.transform(feats))
        preds_all[axis] = full
        chosen[axis] = kind
        print(f"  -> chose {kind} (cv_rho {cv_rho:.3f})")
        report("TEST", y_te[:, a], full[te_idx], te_cov)

        if kind == "ridge_genome":
            w = model.coef_
            order = np.argsort(w)
            top = [tag_names[i] for i in order[-8:][::-1]]
            bot = [tag_names[i] for i in order[:8]]
            print(f"  learned top tags +: {', '.join(top)}")
            print(f"  learned top tags -: {', '.join(bot)}")
        print()

    scores = np.stack([preds_all[a] for a in AXIS_NAMES], axis=1)
    print("predicted label-scale distribution (1-10):")
    for a, axis in enumerate(AXIS_NAMES):
        c = scores[:, a]
        print(f"  {axis:9} mean {c.mean():.2f}  sd {c.std():.2f}  "
              f"min {c.min():.2f}  max {c.max():.2f}")

    # Map onto the front end's [-1, 1] with an affine stretch from the 1st to
    # 99th percentile. Deliberately NOT a rank transform: ranking would force
    # every axis flat, which is what made the old cloud a uniform cube with no
    # clusters or empty regions. An affine map preserves shape, skew and gaps
    # exactly and only changes the units, while still using the full volume —
    # a raw 1-10 mapping would leave everything bunched near the middle,
    # since real movies aren't spread evenly across these scales.
    out = np.empty_like(scores)
    for a in range(scores.shape[1]):
        lo, hi = np.percentile(scores[:, a], [1, 99])
        out[:, a] = np.clip((scores[:, a] - lo) / (hi - lo), 0, 1) * 2 - 1
    np.save(args.out, out.astype(np.float32))
    print(f"\nWrote {out.shape} -> {args.out}  (models: {chosen})")
    print("display spread after stretch (fraction of movies per axis third):")
    for a, axis in enumerate(AXIS_NAMES):
        c = out[:, a]
        thirds = [float((c < -1 / 3).mean()), float(((c >= -1 / 3) & (c < 1 / 3)).mean()),
                  float((c >= 1 / 3).mean())]
        print(f"  {axis:9} low {thirds[0]:.0%}  mid {thirds[1]:.0%}  high {thirds[2]:.0%}")


if __name__ == "__main__":
    main()
