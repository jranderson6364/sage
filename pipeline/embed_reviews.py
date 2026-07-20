"""Embed TMDB user reviews for the semantic-axis "review" signal.

Each movie's reviews are embedded individually (same model as embed_text.py)
then mean-pooled and re-normalized into one vector per movie. Averaging the
raw text instead would silently drop everything past the model's ~256-token
window for a concatenated blob; embedding each review separately lets every
review contribute regardless of how many there are.

Reads pipeline/data/reviews.json + movies.parquet, writes
pipeline/data/review_emb.npy (float32, one row per covered movie) and
pipeline/data/review_rows.json (row indices into movies.parquet, same
coverage-subset pattern as build_genome.py's genome_rows.json).

Usage:
    python embed_reviews.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
MODEL = "all-MiniLM-L6-v2"  # must match embed_text.py / semantic_axes.py


def main() -> None:
    movies = pd.read_parquet(DATA_DIR / "movies.parquet")
    reviews = json.loads((DATA_DIR / "reviews.json").read_text())

    from sentence_transformers import SentenceTransformer  # slow import

    model = SentenceTransformer(MODEL)

    tmdb_to_row = {int(tid): i for i, tid in enumerate(movies["tmdb_id"])}
    rows: list[int] = []
    all_texts: list[str] = []
    splits = [0]
    for tmdb_id, texts in reviews.items():
        row = tmdb_to_row.get(int(tmdb_id))
        if row is None or not texts:
            continue
        rows.append(row)
        all_texts.extend(texts)
        splits.append(len(all_texts))

    print(f"{len(rows)}/{len(movies)} movies covered, {len(all_texts)} reviews to embed")

    flat_emb = model.encode(
        all_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    ).astype(np.float32)

    pooled = np.empty((len(rows), flat_emb.shape[1]), dtype=np.float32)
    for i in range(len(rows)):
        chunk = flat_emb[splits[i]: splits[i + 1]]
        v = chunk.mean(axis=0)
        pooled[i] = v / np.linalg.norm(v)

    np.save(DATA_DIR / "review_emb.npy", pooled)
    (DATA_DIR / "review_rows.json").write_text(json.dumps(rows))
    print(f"Wrote {pooled.shape} review embeddings -> data/review_emb.npy")


if __name__ == "__main__":
    main()
