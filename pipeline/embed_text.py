"""Embed movie text for the "similar story" lens.

Builds one document per movie from overview + tagline + keywords + genres
(deliberately no cast/director — those measure star overlap, not story) and
encodes with a sentence-transformers model. Embeddings are L2-normalized so
downstream cosine similarity is a plain dot product.

Reads pipeline/data/movies.parquet, writes pipeline/data/text_emb.npy
(float32, one row per movie, same order as the parquet).

Usage:
    python embed_text.py
    python embed_text.py --model all-mpnet-base-v2   # slower, higher quality
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


def build_doc(row: pd.Series) -> str:
    parts = []
    if row["overview"]:
        parts.append(row["overview"])
    if row["tagline"]:
        parts.append(row["tagline"])
    if len(row["keywords"]):
        parts.append("Keywords: " + ", ".join(row["keywords"]) + ".")
    if len(row["genres"]):
        parts.append("Genres: " + ", ".join(row["genres"]) + ".")
    return " ".join(parts) or str(row["title"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    movies = pd.read_parquet(DATA_DIR / "movies.parquet")
    docs = [build_doc(row) for _, row in movies.iterrows()]
    print(f"{len(docs)} docs, median length "
          f"{int(np.median([len(d) for d in docs]))} chars")

    from sentence_transformers import SentenceTransformer  # slow import

    model = SentenceTransformer(args.model)
    emb = model.encode(
        docs,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    out = DATA_DIR / "text_emb.npy"
    np.save(out, emb)
    print(f"Wrote {emb.shape} embeddings -> {out}")


if __name__ == "__main__":
    main()
