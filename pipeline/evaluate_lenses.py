"""Score the content-similarity channels against audience ground truth.

Ground truth: adjusted-cosine item-item similarity from real co-ratings
(user-mean-centered, movies with >= --min-ratings). If people who loved A
also loved B, A and B are "truly" related — the best proxy we have for
recommendation accuracy without running an A/B test.

Channels evaluated (all content-only; ALS is excluded because it is *trained*
on the ratings the truth is built from):

  - text     story embeddings (text_emb.npy)
  - genome   tag-genome relevance vectors (genome.npy)
  - tfidf    TF-IDF over TMDB keyword lists
  - blend    w * channels, weights grid-searched on half the movies,
             reported on the held-out half

Metric: Recall@10-in-50 — of a channel's top-10 neighbors, how many sit in
the truth's top-50 — averaged over movies covered by every channel.

Usage:
    python evaluate_lenses.py
"""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

DATA_DIR = Path(__file__).parent / "data"
K, TRUTH_K = 10, 50


def truth_similarity(min_ratings: int) -> tuple[np.ndarray, list[int]]:
    """Adjusted-cosine item-item sim; returns (matrix, movie row indices)."""
    movies = pd.read_parquet(DATA_DIR / "movies.parquet")
    ratings = pd.read_parquet(DATA_DIR / "ratings.parquet")
    ratings["rating"] -= ratings.groupby("userId")["rating"].transform("mean")

    counts = ratings["movieId"].value_counts()
    keep = counts[counts >= min_ratings].index
    ratings = ratings[ratings["movieId"].isin(keep)]

    ml_to_row = {
        int(ml): i for i, ml in movies["movielens_id"].dropna().astype(int).items()
    }
    ratings = ratings[ratings["movieId"].isin(ml_to_row)]

    m_cat = ratings["movieId"].astype("category")
    u_cat = ratings["userId"].astype("category")
    mat = csr_matrix(
        (ratings["rating"].astype(np.float32),
         (m_cat.cat.codes, u_cat.cat.codes)),
    )
    norms = np.sqrt(mat.multiply(mat).sum(axis=1)).A.ravel()
    mat = mat.multiply(1 / np.maximum(norms, 1e-12)[:, None]).tocsr()
    sim = (mat @ mat.T).toarray()
    np.fill_diagonal(sim, -np.inf)

    rows = [ml_to_row[int(ml)] for ml in m_cat.cat.categories]
    return sim, rows


def keyword_tfidf(movies: pd.DataFrame) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(analyzer=lambda kws: list(kws), min_df=2)
    x = vec.fit_transform(movies["keywords"])
    return x  # sparse, L2-normalized by default


def rank_normalize(sim: np.ndarray) -> np.ndarray:
    """Per-row rank percentile in [0,1], so channels blend on equal footing."""
    order = np.argsort(np.argsort(sim, axis=1), axis=1)
    return order.astype(np.float32) / (sim.shape[1] - 1)


def recall(sim: np.ndarray, truth_top: np.ndarray, eligible: np.ndarray) -> float:
    hits = 0
    for i in eligible:
        top = np.argpartition(-sim[i], K)[:K]
        hits += len(set(top) & set(truth_top[i]))
    return hits / (len(eligible) * K)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-ratings", type=int, default=50)
    args = parser.parse_args()

    movies = pd.read_parquet(DATA_DIR / "movies.parquet")
    n = len(movies)

    truth, truth_rows = truth_similarity(args.min_ratings)
    print(f"truth: {len(truth_rows)} movies with >= {args.min_ratings} ratings")

    # Restrict everything to the truth subset, in truth order.
    sub = np.array(truth_rows)
    emb = np.load(DATA_DIR / "text_emb.npy")[sub]
    text_sim = emb @ emb.T

    genome = np.load(DATA_DIR / "genome.npy")
    genome_rows = json.loads((DATA_DIR / "genome_rows.json").read_text())
    g_map = {r: i for i, r in enumerate(genome_rows)}
    covered = np.array([r in g_map for r in sub])
    g_idx = np.array([g_map.get(r, 0) for r in sub])
    genome_sim = genome[g_idx] @ genome[g_idx].T

    tfidf = keyword_tfidf(movies)[sub]
    tfidf_sim = (tfidf @ tfidf.T).toarray()

    for s in (text_sim, genome_sim, tfidf_sim):
        np.fill_diagonal(s, -np.inf)

    truth_top = np.argpartition(-truth, TRUTH_K, axis=1)[:, :TRUTH_K]
    eligible = np.where(covered)[0]
    rng = np.random.default_rng(42)
    rng.shuffle(eligible)
    fit, test = eligible[: len(eligible) // 2], eligible[len(eligible) // 2:]

    print(f"\n{'channel':<10} recall@10-in-50 (n={len(test)} held-out movies)")
    print(f"{'random':<10} {TRUTH_K / len(sub):.3f}")
    for name, s in [("text", text_sim), ("genome", genome_sim), ("tfidf", tfidf_sim)]:
        print(f"{name:<10} {recall(s, truth_top, test):.3f}")

    # Blend: grid-search simplex weights on the fit half, report on test half.
    ranks = {name: rank_normalize(s) for name, s in
             [("text", text_sim), ("genome", genome_sim), ("tfidf", tfidf_sim)]}
    best = (0.0, None)
    for wt, wg in product(np.arange(0, 1.01, 0.1), repeat=2):
        if wt + wg > 1:
            continue
        wk = 1 - wt - wg
        s = wt * ranks["text"] + wg * ranks["genome"] + wk * ranks["tfidf"]
        r = recall(s, truth_top, fit)
        if r > best[0]:
            best = (r, (round(wt, 1), round(wg, 1), round(wk, 1)))
    wt, wg, wk = best[1]
    s = wt * ranks["text"] + wg * ranks["genome"] + wk * ranks["tfidf"]
    print(f"{'blend':<10} {recall(s, truth_top, test):.3f}   "
          f"(text {wt} · genome {wg} · tfidf {wk}, fit on other half)")

    # The master lens as shipped in export_web.py: weighted RRF over each
    # channel's top-50. Includes ALS, which shares training data with the
    # truth — read this row as an optimistic upper bound, not a fair
    # content-only comparison.
    als_f = np.load(DATA_DIR / "als_item_factors.npy")
    als_ids = json.loads((DATA_DIR / "als_movielens_ids.json").read_text())
    a_map = {ml: i for i, ml in enumerate(als_ids)}
    sub_ml = [
        int(movies["movielens_id"].iat[r])
        if pd.notna(movies["movielens_id"].iat[r]) else None
        for r in sub
    ]
    a_cov = np.array([ml in a_map if ml is not None else False for ml in sub_ml])
    a_idx = np.array([a_map.get(ml, 0) if ml is not None else 0 for ml in sub_ml])
    als_sim = als_f[a_idx] @ als_f[a_idx].T
    np.fill_diagonal(als_sim, -np.inf)

    fuse_k, rrf_k = 50, 60
    weights = {"text": 0.2, "genome": 0.5, "als": 0.3}
    tops = {
        "text": np.argsort(-text_sim, axis=1)[:, :fuse_k],
        "genome": np.argsort(-genome_sim, axis=1)[:, :fuse_k],
        "als": np.argsort(-als_sim, axis=1)[:, :fuse_k],
    }
    hits = 0
    for i in test:
        scores: dict[int, float] = {}
        for ch, top in tops.items():
            if ch == "als" and not a_cov[i]:
                continue
            for r, j in enumerate(top[i]):
                scores[j] = scores.get(j, 0.0) + weights[ch] / (rrf_k + r + 1)
        top10 = sorted(scores, key=lambda j: -scores[j])[:K]
        hits += len(set(top10) & set(truth_top[i]))
    print(f"{'master':<10} {hits / (len(test) * K):.3f}   "
          f"(shipped RRF, text 0.2 · genome 0.5 · als 0.3; ALS shares data "
          f"with truth)")


if __name__ == "__main__":
    main()
