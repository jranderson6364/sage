"""Project text embeddings to a 2D map layout with UMAP.

The map is laid out from the *story* embeddings for every movie (the audience
lens only covers movies with enough ratings, and node positions shouldn't
change when the user toggles lenses).

Reads pipeline/data/text_emb.npy, writes pipeline/data/layout.npy
(float32 Nx2, same row order), rescaled to [-1, 1] per axis.

Usage:
    python layout_umap.py
    python layout_umap.py --n-neighbors 15 --min-dist 0.05   # tighter clusters
"""

import argparse
import time
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-neighbors", type=int, default=30,
                        help="higher = more global structure")
    parser.add_argument("--min-dist", type=float, default=0.1,
                        help="lower = tighter clusters")
    args = parser.parse_args()

    emb = np.load(DATA_DIR / "text_emb.npy")
    print(f"UMAP on {emb.shape} embeddings "
          f"(n_neighbors={args.n_neighbors}, min_dist={args.min_dist})")

    import umap  # slow import

    start = time.time()
    xy = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="cosine",
        random_state=42,
    ).fit_transform(emb)
    print(f"done in {time.time() - start:.0f}s")

    # Rescale to [-1, 1] so the front end never cares about UMAP's raw range.
    xy = xy.astype(np.float32)
    xy -= xy.min(axis=0)
    xy /= xy.max(axis=0)
    xy = xy * 2 - 1

    out = DATA_DIR / "layout.npy"
    np.save(out, xy)
    print(f"Wrote {xy.shape} layout -> {out}")


if __name__ == "__main__":
    main()
