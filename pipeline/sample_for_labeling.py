"""Draw a stratified sample of movies to hand-label for axis validation.

Deliberately prints NO axis scores — labels have to be formed independently
of what the current scorer thinks, or the "validation" set just ratifies it.

Genome-uncovered movies are oversampled far beyond their ~20% share of the
catalog: they're the group most exposed to axis error (no community tags to
anchor them) and the only group an imputation fix would move, so a
proportional draw would leave too few to measure a change against.

Usage:
    python sample_for_labeling.py > sample.txt
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

N_COVERED = 60
N_UNCOVERED = 60
SEED = 20260721
# Hand-labeling needs real familiarity with the film; below this many TMDB
# votes it's mostly guessing from the overview, which would just re-derive
# the text signal the eval is supposed to judge.
MIN_VOTES = 900


def main() -> None:
    movies = pd.read_parquet(DATA_DIR / "movies.parquet")
    genome_rows = set(json.loads((DATA_DIR / "genome_rows.json").read_text()))

    eligible = movies.index[movies["vote_count"] >= MIN_VOTES]
    covered = [i for i in eligible if i in genome_rows]
    uncovered = [i for i in eligible if i not in genome_rows]
    print(f"# eligible (>={MIN_VOTES} votes): "
          f"{len(covered)} genome-covered, {len(uncovered)} uncovered")

    rng = np.random.default_rng(SEED)
    pick = list(rng.choice(covered, min(N_COVERED, len(covered)), replace=False)) + \
        list(rng.choice(uncovered, min(N_UNCOVERED, len(uncovered)), replace=False))

    for i in sorted(int(x) for x in pick):
        row = movies.loc[i]
        cov = "G" if i in genome_rows else "-"
        overview = (row["overview"] or "")[:110].replace("\n", " ")
        print(f"{i}\t{cov}\t{row['title']} ({row['year']})\t"
              f"{'/'.join(row['genres'][:3])}\t{overview}")


if __name__ == "__main__":
    main()
