"""Export the static JSON that powers the web app.

Combines movies.parquet with the semantic axes and every similarity space
into one file. The app has a single recommendation model, so exactly one
neighbor list ships per movie:

  - nn: weighted reciprocal-rank fusion of three internal channels — story
        embedding (all movies), tag genome (covered subset), and ALS item
        factors (rated subset). Never empty: a movie missing the genome/ALS
        channels still fuses whatever it does have, down to text alone.

The individual channels are deliberately *not* exported — they exist only to
feed the fusion. Movie positions come from the axes (levity/threat/intimacy),
so no 2D layout is exported either.

Each movie also carries its top genome tags ([tag_index, relevance] pairs into
the top-level "tags" name list) so the UI can explain *why* two movies match
and drive aspect steering. Neighbor lists hold row indices into the movies
array.

Writes web/public/data/movies.json (Vite serves public/ verbatim).

Usage:
    python export_web.py
    python export_web.py --k 15
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_DIR = Path(__file__).parent
DATA_DIR = PIPELINE_DIR / "data"
WEB_DATA_DIR = PIPELINE_DIR.parent / "web" / "public" / "data"

OVERVIEW_CHARS = 300

# The recommendation model: reciprocal-rank fusion over each channel's
# top-FUSE_K. Weights follow evaluate_lenses.py (genome is by far the
# strongest content signal; ALS adds real audience taste; text is the
# always-available floor that carries movies the other channels don't
# cover). Refine here, then re-run evaluate_lenses.py to check the change
# actually helped.
FUSE_K = 50
RRF_K = 60
MASTER_WEIGHTS = {"genome": 0.5, "als": 0.3, "text": 0.2}


def top_k_cosine(emb: np.ndarray, k: int) -> np.ndarray:
    """Row indices of the k nearest neighbors (self excluded), rows normalized."""
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    # argpartition doesn't order the top k; sort them by similarity.
    rows = np.arange(len(sim))[:, None]
    order = np.argsort(-sim[rows, idx], axis=1)
    return idx[rows, order]


def truncate(text: str | None, limit: int) -> str:
    if not text or len(text) <= limit:
        return text or ""
    return text[:limit].rsplit(" ", 1)[0] + "…"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10, help="neighbors per movie")
    parser.add_argument("--axes", default=None,
                        help="axis scores to ship (default: learned, else hand-tuned)")
    args = parser.parse_args()

    movies = pd.read_parquet(DATA_DIR / "movies.parquet")
    text_emb = np.load(DATA_DIR / "text_emb.npy")
    # Prefer the fitted axes (train_axes.py) over the hand-tuned scorer;
    # semantic_axes.py's output stays the fallback and the comparison point.
    axes_path = Path(args.axes) if args.axes else (
        DATA_DIR / "axes_learned.npy" if (DATA_DIR / "axes_learned.npy").exists()
        else DATA_DIR / "axes.npy")
    axes = np.load(axes_path)  # levity, threat, intimacy
    print(f"axes: {axes_path.name}")
    assert len(movies) == len(text_emb) == len(axes)

    nn_text = top_k_cosine(text_emb, FUSE_K)

    # Genome channel: k-NN in tag-genome space, over the covered subset.
    genome = np.load(DATA_DIR / "genome.npy")
    genome_rows = json.loads((DATA_DIR / "genome_rows.json").read_text())
    tag_names = json.loads((DATA_DIR / "genome_tags.json").read_text())
    top_tags = json.loads((DATA_DIR / "genome_top_tags.json").read_text())
    nn_g = top_k_cosine(genome, min(FUSE_K, len(genome_rows) - 1))
    nn_vibe = {
        movie_i: [genome_rows[j] for j in nn_g[gi]]
        for gi, movie_i in enumerate(genome_rows)
    }
    movie_tags = {movie_i: top_tags[gi] for gi, movie_i in enumerate(genome_rows)}
    print(f"genome channel covers {len(genome_rows)} / {len(movies)} movies")

    # ALS covers the subset of movies that had enough MovieLens ratings.
    factors = np.load(DATA_DIR / "als_item_factors.npy")
    als_ids = json.loads((DATA_DIR / "als_movielens_ids.json").read_text())
    factor_row = {ml_id: i for i, ml_id in enumerate(als_ids)}

    covered = [
        i for i, ml_id in enumerate(movies["movielens_id"])
        if pd.notna(ml_id) and int(ml_id) in factor_row
    ]
    sub = factors[[factor_row[int(movies["movielens_id"].iat[i])] for i in covered]]
    nn_sub = top_k_cosine(sub, min(FUSE_K, len(covered) - 1))
    nn_als: dict[int, list[int]] = {
        movie_i: [covered[j] for j in nn_sub[si]]
        for si, movie_i in enumerate(covered)
    }
    print(f"ALS channel covers {len(covered)} / {len(movies)} movies")

    def fuse(i: int) -> list[int]:
        """The recommendation: weighted RRF over the available channels."""
        lists = {
            "text": nn_text[i],
            "genome": nn_vibe.get(i),
            "als": nn_als.get(i),
        }
        scores: dict[int, float] = {}
        for channel, ranked in lists.items():
            if ranked is None:
                continue
            w = MASTER_WEIGHTS[channel]
            for r, j in enumerate(ranked):
                scores[int(j)] = scores.get(int(j), 0.0) + w / (RRF_K + r + 1)
        top = sorted(scores.items(), key=lambda kv: -kv[1])[: args.k]
        return [j for j, _ in top]

    nodes = []
    for i, row in movies.iterrows():
        nodes.append({
            "title": row["title"],
            "year": int(row["year"]) if row["year"] else None,
            "genres": list(row["genres"]),
            "directors": list(row["directors"]),
            "overview": truncate(row["overview"], OVERVIEW_CHARS),
            "poster": row["poster_path"] if pd.notna(row["poster_path"]) else None,
            "rating": round(float(row["vote_average"]), 1),
            "votes": int(row["vote_count"]),
            "levity": round(float(axes[i, 0]), 3),
            "threat": round(float(axes[i, 1]), 3),
            "intimacy": round(float(axes[i, 2]), 3),
            "nn": fuse(i),
            "tags": movie_tags.get(i, []),
        })

    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Quantized genome for client-side aspect steering: the front end
    # re-ranks recommendations after boosting user-picked tag dimensions.
    # Layout: uint32 n, uint32 dim, float32 scales[n], uint8 data[n*dim]
    # where value = data/255 * scale (rows are the L2-normalized genome).
    scale = genome.max(axis=1)
    q = np.round(genome / np.maximum(scale[:, None], 1e-12) * 255).astype(np.uint8)
    bin_out = WEB_DATA_DIR / "genome_q8.bin"
    with open(bin_out, "wb") as f:
        f.write(np.array([len(q), q.shape[1]], dtype=np.uint32).tobytes())
        f.write(scale.astype(np.float32).tobytes())
        f.write(q.tobytes())
    print(f"Wrote steering matrix -> {bin_out} ({bin_out.stat().st_size / 1e6:.1f} MB)")

    out = WEB_DATA_DIR / "movies.json"
    out.write_text(
        # allow_nan=False: a stray pandas NaN would otherwise serialize as
        # bare NaN, which is invalid JSON and breaks the whole front end.
        json.dumps({
            "movies": nodes,
            "tags": tag_names,
            "genome_rows": genome_rows,
        }, separators=(",", ":"), ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(nodes)} movies -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
