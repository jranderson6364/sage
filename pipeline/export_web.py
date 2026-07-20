"""Export the static JSON that powers the web app.

Combines movies.parquet, the UMAP layout, and all similarity spaces into one
file. Neighbors are precomputed here so the front end does zero math:

  - nn_best: master recommendation — weighted reciprocal-rank fusion of the
             three channels below (never empty; falls back to whatever
             channels cover the movie)
  - nn_text: k nearest by story embedding (all movies)
  - nn_vibe: k nearest by tag-genome relevance (empty list if the movie isn't
             in the genome)
  - nn_als:  k nearest by ALS item factors (empty list if the movie didn't
             have enough ratings for the audience lens)

Each movie also carries its top genome tags ([tag_index, relevance] pairs into
the top-level "tags" name list) so the UI can explain *why* two movies match.
Neighbor lists hold row indices into the same movies array.

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

# Master lens: reciprocal-rank fusion over each channel's top-FUSE_K.
# Weights follow evaluate_lenses.py (genome is by far the strongest content
# signal; ALS adds real audience taste; text is the always-available floor
# that carries movies the other channels don't cover). Refine here, then
# re-run evaluate_lenses.py to check the change actually helped.
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


def galaxy_layout(layout: np.ndarray, years: pd.Series) -> tuple[np.ndarray, list[dict]]:
    """Polar "galaxy" projection of the map.

    Angle keeps each movie's story neighborhood (bearing from the UMAP
    centroid), radius is release year — oldest at the core, newest at the rim,
    like tree rings. Radius follows the cumulative movie count (sqrt for equal
    area density) rather than the year itself, so sparse early decades compress
    toward the core instead of leaving empty rings.

    Returns per-movie xy plus ring positions for decade labels.
    """
    R0 = 0.06  # hole at the core so the oldest films don't stack on one point

    cx, cy = layout.mean(axis=0)
    theta = np.arctan2(layout[:, 1] - cy, layout[:, 0] - cx)

    yr = pd.to_numeric(years, errors="coerce")
    yr = yr.fillna(yr.median()).to_numpy(dtype=float)
    n = len(yr)
    rank = np.empty(n)
    rank[np.argsort(yr, kind="stable")] = np.arange(n)
    radius = R0 + (1 - R0) * np.sqrt((rank + 0.5) / n)

    xy = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)

    decades = []
    for d in range(1930, 2030, 10):
        frac_before = float((yr < d).mean())
        if 0.01 < frac_before < 0.995:
            decades.append({
                "label": f"{d}s",
                "r": round(R0 + (1 - R0) * np.sqrt(frac_before), 4),
            })
    return xy.astype(np.float32), decades


def truncate(text: str | None, limit: int) -> str:
    if not text or len(text) <= limit:
        return text or ""
    return text[:limit].rsplit(" ", 1)[0] + "…"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10, help="neighbors per lens")
    args = parser.parse_args()

    movies = pd.read_parquet(DATA_DIR / "movies.parquet")
    layout = np.load(DATA_DIR / "layout.npy")
    text_emb = np.load(DATA_DIR / "text_emb.npy")
    axes = np.load(DATA_DIR / "axes.npy")  # levity, threat, intimacy (semantic_axes.py)
    assert len(movies) == len(layout) == len(text_emb) == len(axes)

    nn_text = top_k_cosine(text_emb, FUSE_K)
    galaxy, decades = galaxy_layout(layout, movies["year"])

    # Vibe lens: k-NN in tag-genome space, over the covered subset.
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
    print(f"vibe lens covers {len(genome_rows)} / {len(movies)} movies")

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
    print(f"audience lens covers {len(covered)} / {len(movies)} movies")

    def fuse(i: int) -> list[int]:
        """Master recommendation: weighted RRF over the available channels."""
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
            "x": round(float(layout[i, 0]), 4),
            "y": round(float(layout[i, 1]), 4),
            "gx": round(float(galaxy[i, 0]), 4),
            "gy": round(float(galaxy[i, 1]), 4),
            "levity": round(float(axes[i, 0]), 3),
            "threat": round(float(axes[i, 1]), 3),
            "intimacy": round(float(axes[i, 2]), 3),
            "nn_best": fuse(i),
            "nn_text": [int(j) for j in nn_text[i][: args.k]],
            "nn_vibe": nn_vibe.get(i, [])[: args.k],
            "nn_als": nn_als.get(i, [])[: args.k],
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
            "decades": decades,
            "tags": tag_names,
            "genome_rows": genome_rows,
        }, separators=(",", ":"), ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(nodes)} movies -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
