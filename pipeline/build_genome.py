"""Build the "similar vibe" channel from the MovieLens Tag Genome.

The genome scores ~13k movies against 1,128 community tags (atmospheric,
slow burn, twist ending, ...) — dense relevance in [0, 1], computed by
GroupLens from user tags and reviews. It captures how movies *feel* in a way
plot text can't, which makes it the essence-similarity channel.

Reads movies.parquet (needs movielens_id from join_movielens.py) plus
genome-scores.csv / genome-tags.csv, writes:

  - genome.npy            (float32, one L2-normalized row per *covered* movie)
  - genome_rows.json      (movie row index in movies.parquet per genome row)
  - genome_tags.json      (1,128 tag names, column order)
  - genome_top_tags.json  (per covered movie: top-10 [tag_index, relevance],
                           raw relevance for display and shared-tag overlap)

Usage:
    python build_genome.py
    python build_genome.py --ml-dir data/ml-25m
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml-dir", type=Path, default=DATA_DIR / "ml-25m")
    args = parser.parse_args()

    movies = pd.read_parquet(DATA_DIR / "movies.parquet")
    tags = pd.read_csv(args.ml_dir / "genome-tags.csv")
    scores = pd.read_csv(args.ml_dir / "genome-scores.csv")
    n_tags = len(tags)

    # Genome rows for our movies only, in movies.parquet row order.
    wanted = movies["movielens_id"].dropna().astype(int)
    covered_ml = set(scores["movieId"].unique()) & set(wanted)
    rows = [i for i, ml in zip(wanted.index, wanted) if ml in covered_ml]
    ml_ids = [int(movies["movielens_id"].iat[i]) for i in rows]
    print(f"genome covers {len(rows)} / {len(movies)} movies "
          f"({n_tags} tags)")

    scores = scores[scores["movieId"].isin(covered_ml)]
    mat = scores.pivot(index="movieId", columns="tagId", values="relevance")
    # copy=True: to_numpy can hand back a read-only view of the frame,
    # which the in-place normalization below can't write to.
    mat = mat.reindex(index=ml_ids).to_numpy(dtype=np.float32, copy=True)
    assert not np.isnan(mat).any(), "genome matrix has holes"

    top_idx = np.argsort(-mat, axis=1)[:, :10]
    tops = [
        [[int(t), round(float(mat[r, t]), 2)] for t in top_idx[r]]
        for r in range(len(rows))
    ]
    (DATA_DIR / "genome_top_tags.json").write_text(json.dumps(tops))

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat /= np.maximum(norms, 1e-12)

    np.save(DATA_DIR / "genome.npy", mat)
    (DATA_DIR / "genome_rows.json").write_text(json.dumps(rows))
    (DATA_DIR / "genome_tags.json").write_text(
        json.dumps(tags.sort_values("tagId")["tag"].tolist()))
    print(f"Wrote {mat.shape} genome matrix -> data/genome.npy")


if __name__ == "__main__":
    main()
