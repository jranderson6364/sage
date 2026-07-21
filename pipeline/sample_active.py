"""Draw the next batch of movies to hand-label, by information value.

Random sampling wastes labels on films every channel already agrees about.
This ranks candidates by how much the genome / review / text channels
*disagree* on an axis — those are where the current scoring is least certain
and a human label settles the most.

Half the batch is drawn that way; the other half is uniform random, because
a training set made only of hard cases is a biased view of the catalog and
would teach a model the wrong prior.

Prints no computed axis scores (only the disagreement bucket), so labels
stay independent of what the scorer currently believes.

Usage:
    python sample_active.py --n 300 > batch.txt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_DIR = Path(__file__).parent
DATA_DIR = PIPELINE_DIR / "data"
SEED = 20260721


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--min-votes", type=int, default=1200,
                        help="familiarity floor — below this, labeling is guesswork")
    args = parser.parse_args()

    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(PIPELINE_DIR))
    from semantic_axes import (AXES, GENOME_TAGS, CHANNEL_WEIGHTS, rank01,
                               MODEL, impute_genome)

    movies = pd.read_parquet(DATA_DIR / "movies.parquet")
    emb = np.load(DATA_DIR / "text_emb.npy")
    genome = np.load(DATA_DIR / "genome.npy")
    genome_rows = json.loads((DATA_DIR / "genome_rows.json").read_text())
    tag_names = json.loads((DATA_DIR / "genome_tags.json").read_text())
    tag_idx = {t: i for i, t in enumerate(tag_names)}
    review_emb = np.load(DATA_DIR / "review_emb.npy")
    review_rows = json.loads((DATA_DIR / "review_rows.json").read_text())
    review_pos = {r: i for i, r in enumerate(review_rows)}
    n = len(movies)

    already = set(int(k) for k in
                  json.loads((PIPELINE_DIR / "axis_labels.json").read_text())["labels"])

    genome_all, _ = impute_genome(emb, genome, genome_rows, n, 10, 5.0)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)

    # Per-axis channel disagreement = spread between the three percentiles.
    spread = np.zeros(n)
    for name, (pos, neg) in AXES.items():
        a = model.encode(pos + neg, normalize_embeddings=True)
        d = a[:len(pos)].mean(0) - a[len(pos):].mean(0)
        d /= np.linalg.norm(d)
        text_rank = rank01(emb @ d)
        gp, gn = GENOME_TAGS[name]
        pi = [tag_idx[t] for t in gp if t in tag_idx]
        ni = [tag_idx[t] for t in gn if t in tag_idx]
        raw = genome_all[:, pi].sum(1) - (genome_all[:, ni].sum(1) if ni else 0)
        g_rank = rank01(raw)
        r_sub = rank01(review_emb @ d)
        r_rank = np.full(n, np.nan)
        for row, i in review_pos.items():
            r_rank[row] = r_sub[i]
        stack = np.vstack([text_rank, g_rank, np.nan_to_num(r_rank, nan=np.nan)])
        spread += np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)

    eligible = np.array([
        i for i in range(n)
        if i not in already and movies["vote_count"].iat[i] >= args.min_votes
    ])
    rng = np.random.default_rng(SEED + 1)
    half = args.n // 2
    hard = eligible[np.argsort(-spread[eligible])[:half]]
    rest = np.setdiff1d(eligible, hard)
    rand = rng.choice(rest, min(args.n - half, len(rest)), replace=False)
    pick = np.concatenate([hard, rand])

    print(f"# {len(pick)} to label ({len(hard)} high-disagreement, {len(rand)} random); "
          f"{len(already)} already labeled")
    for i in sorted(int(x) for x in pick):
        row = movies.loc[i]
        bucket = "HARD" if i in set(hard.tolist()) else "rand"
        overview = (row["overview"] or "")[:95].replace("\n", " ")
        print(f"{i}\t{bucket}\t{row['title']} ({row['year']})\t"
              f"{'/'.join(row['genres'][:3])}\t{overview}")


if __name__ == "__main__":
    main()
