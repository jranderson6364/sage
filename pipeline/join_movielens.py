"""Join TMDB movies with MovieLens IDs and export ratings for the ALS lens.

Reads pipeline/data/movies.parquet (from fetch_tmdb.py) and a MovieLens dataset
directory, joins via links.csv (movieId <-> tmdbId), and writes:

  - movies.parquet          (adds movielens_id column, in place)
  - ratings.parquet         (ratings filtered to matched movies only)

Usage:
    python join_movielens.py                          # ml-latest-small
    python join_movielens.py --ml-dir data/ml-25m     # after swapping in ml-25m
"""

import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml-dir", type=Path, default=DATA_DIR / "ml-latest-small")
    args = parser.parse_args()

    movies = pd.read_parquet(DATA_DIR / "movies.parquet")
    links = pd.read_csv(args.ml_dir / "links.csv", dtype={"tmdbId": "Int64"})

    # A handful of TMDB ids appear under multiple MovieLens entries (re-releases,
    # data errors); keep the lowest movieId, which is the older canonical entry.
    id_map = (
        links.dropna(subset=["tmdbId"])
        .sort_values("movieId")
        .drop_duplicates("tmdbId")
        .set_index("tmdbId")["movieId"]
    )
    movies["movielens_id"] = movies["tmdb_id"].map(id_map).astype("Int64")

    matched = movies["movielens_id"].notna()
    print(f"Matched {matched.sum()} / {len(movies)} movies "
          f"({matched.mean():.1%}) to MovieLens IDs")

    movies.to_parquet(DATA_DIR / "movies.parquet", index=False)

    ratings = pd.read_csv(args.ml_dir / "ratings.csv")
    ratings = ratings[ratings["movieId"].isin(set(movies["movielens_id"].dropna()))]
    ratings.to_parquet(DATA_DIR / "ratings.parquet", index=False)
    print(f"Wrote {len(ratings)} ratings from {ratings['userId'].nunique()} users "
          f"covering {ratings['movieId'].nunique()} movies -> data/ratings.parquet")


if __name__ == "__main__":
    main()
