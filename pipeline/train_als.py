"""Train ALS item factors for the "similar audience" lens.

Factorizes the MovieLens ratings matrix with implicit's ALS. The rating value
is used directly as the confidence weight — for item-item similarity we only
care about the direction of item factors, and this keeps "watched and loved"
weighted above "watched and shrugged".

Items with too few ratings get junk factors, so anything under --min-ratings
is dropped; the web export just won't offer an audience lens for those movies.

Reads pipeline/data/ratings.parquet, writes:
  - als_item_factors.npy    (float32, L2-normalized rows)
  - als_movielens_ids.json  (movieId per factor row, same order)

Usage:
    python train_als.py
    python train_als.py --factors 128 --min-ratings 10   # for ml-25m later
"""

import argparse
import json
import os
from pathlib import Path

# implicit's ALS runs its own threadpool; a threaded BLAS underneath it causes
# heavy contention (implicit warns about exactly this). Must be set pre-import.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

DATA_DIR = Path(__file__).parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--regularization", type=float, default=0.05)
    parser.add_argument("--min-ratings", type=int, default=5,
                        help="drop movies with fewer ratings than this")
    args = parser.parse_args()

    ratings = pd.read_parquet(DATA_DIR / "ratings.parquet")
    counts = ratings["movieId"].value_counts()
    keep = counts[counts >= args.min_ratings].index
    ratings = ratings[ratings["movieId"].isin(keep)]
    print(f"{len(ratings)} ratings on {ratings['movieId'].nunique()} movies "
          f"(>= {args.min_ratings} ratings) from {ratings['userId'].nunique()} users")

    user_ids = ratings["userId"].astype("category")
    movie_ids = ratings["movieId"].astype("category")
    user_items = csr_matrix(
        (
            ratings["rating"].astype(np.float32),
            (user_ids.cat.codes, movie_ids.cat.codes),
        ),
        shape=(user_ids.cat.categories.size, movie_ids.cat.categories.size),
    )

    from implicit.als import AlternatingLeastSquares  # slow import

    model = AlternatingLeastSquares(
        factors=args.factors,
        regularization=args.regularization,
        iterations=args.iterations,
        random_state=42,
    )
    model.fit(user_items)

    factors = np.asarray(model.item_factors, dtype=np.float32)
    norms = np.linalg.norm(factors, axis=1, keepdims=True)
    factors /= np.maximum(norms, 1e-12)

    np.save(DATA_DIR / "als_item_factors.npy", factors)
    ids = [int(m) for m in movie_ids.cat.categories]
    (DATA_DIR / "als_movielens_ids.json").write_text(json.dumps(ids))
    print(f"Wrote {factors.shape} item factors -> data/als_item_factors.npy")


if __name__ == "__main__":
    main()
