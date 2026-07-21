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

# Feature block per axis. Cross-validation genuinely cannot separate these at
# n=279 — every option sits inside one standard error — so these come from a
# mechanistic audit against 100 popular films (evaluate_popular.py), where the
# differences are large and consistent:
#
#   levity    g+rev   reviews say "hilarious" outright; plot text only gives
#                     premise, and premise misleads (The Notebook's summary
#                     reads romantic-serious). MAE 6.3 vs 6.7 genome, 7.6 +text.
#   threat    genome  tags already encode tension precisely (tense, suspense,
#                     terror). Both embeddings only add noise: 9.3 -> 10.3.
#   intimacy  all     tags rarely record "is this film *about* closeness", so
#                     films like The Martian had no signal at all and reverted
#                     to the population mean, which ranking then dressed up as
#                     ~58th percentile. Both embeddings describe connection and
#                     isolation directly: MAE 9.8 -> 7.5, bias +5.9 -> +1.6.
#
# Set to None to fall back to the 1-SE automatic choice.
AXIS_FEATURES = {
    "levity": "ridge_g+rev",
    "threat": "ridge_genome",
    "intimacy": "ridge_all",
}


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
    parser.add_argument("--with-subs", action="store_true",
                        help="include subtitle features (measured not to help)")
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

    # Tried and rejected: dropping format/medium tags (animation, cartoon,
    # pixar, cgi, franchise...) on the theory that they were suppressing threat
    # for animated films, since "cartoon" dominates their tag mass. Measured
    # no effect — popular-100 threat bias -7.0 -> -7.1, MAE 9.13 -> 9.28, The
    # Lion King 21 -> 23. Removing a misleading tag doesn't supply the missing
    # one; those films still carry no positive tension signal either way.
    genome_all = knn_impute(emb, genome, genome_rows, n)
    review_all = knn_impute(emb, review_emb, review_rows, n)

    # TMDB keywords. Unlike the tag genome these exist for *every* movie,
    # including releases far too recent for MovieLens to have tagged — and
    # they're often explicitly tone-bearing ("supernatural horror",
    # "psychological", "unrequited love"). That matters because the genome is
    # missing for ~991 films, and a genome-only model scores those entirely
    # from an imputed, i.e. fabricated, tag vector.
    from sklearn.feature_extraction.text import TfidfVectorizer
    kw_docs = ["; ".join(k) for k in movies["keywords"]]
    kw_vec = TfidfVectorizer(analyzer=lambda d: d.split("; "), min_df=25)
    keywords = kw_vec.fit_transform(kw_docs).toarray().astype(np.float32)
    print(f"keywords: {keywords.shape[1]} dims (min_df=25), "
          f"{(keywords.any(axis=1)).sum()}/{n} movies non-empty")

    # Feature blocks kept separable rather than one lump, because the two
    # embedding sources behave very differently per axis: reviews describe how
    # a film *felt* ("hilarious", "stressful"), while story text only
    # describes premise — and premise is what misled levity (The Notebook's
    # plot reads romantic-serious, its tags are pure romance, yet a
    # text-inclusive model called it levity 60). Offering "+review" without
    # "+text" lets an axis take the useful half.
    base = np.hstack([genome_all, emb, review_all])
    X = {
        "ridge_genome": genome_all,
        "ridge_g+rev": np.hstack([genome_all, review_all]),
        "ridge_g+txt": np.hstack([genome_all, emb]),
        "ridge_gkw": np.hstack([genome_all, keywords]),
        "ridge_all": base,
        "ridge_allkw": np.hstack([base, keywords]),
        "gbm": base,
    }

    # Subtitle timing features. MEASURED NEGATIVE, off by default: they gain
    # +0.002..0.005 cv_rho but *lose* 0.001..0.005 on the held-out test on
    # every axis — CV selection noise, not signal. With 279 labels, 18 extra
    # features can't demonstrate value, and the genome plus embeddings
    # already carry whatever they'd contribute. Kept behind a flag so the
    # result is reproducible rather than just asserted. (The per-decile arcs
    # they also produce are a separate product feature and don't depend on
    # this being useful for scoring — see narrative_arcs.py.)
    sub_path = DATA_DIR / "sub_features.npy"
    if args.with_subs and sub_path.exists():
        subs = np.load(sub_path)
        sub_rows = json.loads((DATA_DIR / "sub_rows.json").read_text())
        subs = np.nan_to_num(subs, nan=0.0, posinf=0.0, neginf=0.0)
        subs_all = knn_impute(emb, subs, sub_rows, n)
        has_sub = np.zeros((n, 1), dtype=np.float32)
        has_sub[sub_rows] = 1.0  # let the model discount borrowed rows
        X["ridge_subs"] = np.hstack([base, subs_all, has_sub])
        X["gbm_subs"] = X["ridge_subs"]
        print(f"subtitle features: {subs.shape[1]} dims, "
              f"{len(sub_rows)}/{n} movies covered")

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
        if kind.startswith("gbm"):
            return HistGradientBoostingRegressor(
                max_depth=3, max_iter=300, learning_rate=0.06, random_state=SEED)
        return Ridge(alpha=alpha)

    preds_all = {}
    chosen = {}
    for a, axis in enumerate(AXIS_NAMES):
        print(f"[{axis}]")

        # Ordered simplest-first; the 1-SE rule below relies on this.
        candidates = [k for k in ["ridge_genome", "ridge_g+rev", "ridge_g+txt",
                                  "ridge_gkw", "ridge_all", "ridge_allkw",
                                  "gbm", "ridge_subs", "gbm_subs"] if k in X]
        scored = []
        for kind in (candidates if not args.model else [args.model]):
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
            # Per-fold spread, so "better" can be distinguished from noise.
            folds = []
            for tr_i, te_i in cv.split(Xtr):
                m = build(kind, best_a if kind.startswith("ridge") else None)
                m.fit(Xtr[tr_i], y_tr[tr_i, a])
                folds.append(spearmanr(y_tr[te_i, a], m.predict(Xtr[te_i])).statistic)
            se = float(np.std(folds, ddof=1) / np.sqrt(len(folds)))
            print(f"  {tag:22} cv_rho {cv_rho:.3f} ±{se:.3f}")
            scored.append((kind, cv_rho, se, model))

        # 1-SE rule: take the simplest model within one standard error of the
        # best, not the raw argmax. Chasing the argmax at n=279 picked
        # ridge_all over ridge_genome on a 0.003 cv_rho edge, and those 768
        # extra embedding dimensions turned out to inject real noise — it
        # scored *worse* than even the hand-tuned scorer on levity for
        # recognizable films (The Notebook levity 60, Titanic 44), because a
        # spurious embedding direction generalizes badly off-distribution.
        # Tags are a far more honest feature space for this.
        pick = AXIS_FEATURES.get(axis) if not args.model else args.model
        if pick and pick in {s[0] for s in scored}:
            kind, cv_rho, se, model = next(s for s in scored if s[0] == pick)
        else:
            best_rho = max(s[1] for s in scored)
            thresh = best_rho - max(s[2] for s in scored if s[1] == best_rho)
            kind, cv_rho, se, model = next(s for s in scored if s[1] >= thresh)
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

    # Rank-normalize onto [-1, 1].
    #
    # An earlier version shipped the raw predicted rating (affine-stretched)
    # on the theory that a real, skewed distribution is more honest than a
    # forced-uniform one. That was wrong for this product. Ridge shrinks
    # predictions toward the mean, so the interior stayed crowded even after
    # the endpoints were stretched: Die Hard's threat fell 90 -> 67, Heat
    # 93 -> 74, while Titanic rose 15 -> 48. Rank metrics couldn't see any of
    # it — compression preserves order, so rho and a rank-based `sep` both
    # stayed happy while every displayed number drifted to the middle.
    #
    # This view exists to *compare* films: uniform spread is what makes the
    # cloud readable and the filter sliders meaningful, and it makes the
    # readout a true percentile again. The learned model's contribution is a
    # better ordering, which survives the transform intact.
    out = np.empty_like(scores)
    for a in range(scores.shape[1]):
        r = np.empty(len(scores))
        r[np.argsort(scores[:, a], kind="stable")] = np.arange(len(scores))
        out[:, a] = ((r + 0.5) / len(scores)) * 2 - 1
    np.save(args.out, out.astype(np.float32))
    print(f"\nWrote {out.shape} -> {args.out}  (models: {chosen})")


if __name__ == "__main__":
    main()
