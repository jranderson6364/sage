"""Cluster movies by the *shape* of their tension, not just its level.

Every other signal in this project reduces a film to a point. Subtitle
timing gives something no tag or plot summary can: how a film moves. A
thriller that ratchets steadily and one that explodes in the last act score
the same on the threat axis and feel nothing alike.

Reagan et al. (2016) found books reduce to six emotional shapes; the same
was later reported for ~6k movie scripts. This does it on subtitle timing
instead — dialogue density, silence, and distress vocabulary per decile of
runtime — which covers far more films than scripts do.

Method: z-normalize each movie's curve so clustering keys on shape rather
than loudness (an intense film and a quiet one can share an arc), then
k-means, then name each centroid by where its peak falls.

Reads data/sub_arcs.npy (subtitle_features.py), writes data/arcs.json:
per-movie smoothed curve + archetype id, plus the centroids.

Usage:
    python narrative_arcs.py
    python narrative_arcs.py --k 6
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent / "data"
SEED = 20260721


def name_shape(c):
    """Describe a centroid in plain language.

    Named primarily by where the peak sits, since that's what a viewer
    actually feels, with two special shapes checked first because they're
    defined by their profile rather than a single crest.
    """
    n = len(c)
    third = n // 3
    early, mid, late = c[:third].mean(), c[third:2 * third].mean(), c[2 * third:].mean()
    peak = int(np.argmax(c))

    if c.std() < 0.45:
        return "Steady"
    # Count real crests: points above +0.5 separated by a dip below 0. This
    # beats comparing act averages, which called single-peaked shapes double
    # whenever the peak happened to sit near a boundary.
    crests, i = [], 0
    while i < n:
        if c[i] > 0.5:
            j = i
            while j + 1 < n and c[j + 1] > 0:
                j += 1
            crests.append(int(np.argmax(c[i:j + 1])) + i)
            i = j + 1
        else:
            i += 1

    if len(crests) >= 2:
        # Calm middle with tense bookends is its own recognizable shape.
        if mid < early - 0.4 and mid < late - 0.4:
            return "Twin peaks"
        # Otherwise distinguish by where the first crest lands, so two
        # genuinely different double-peaked shapes don't collide into one
        # name and pick up a meaningless numeric suffix in the UI.
        return "Double climax" if crests[0] < n / 3 else "Midpoint + finale"
    if peak <= 1:
        return "Cold open"          # loudest at the start, winds down
    if peak <= 4:
        return "Midpoint spike"
    if peak <= 6:
        return "Third-act climax"
    return "Slow burn"              # builds all the way to the finale


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    arcs = np.load(DATA_DIR / "sub_arcs.npy")   # (m, 3, deciles)
    rows = json.loads((DATA_DIR / "sub_rows.json").read_text())
    movies = pd.read_parquet(DATA_DIR / "movies.parquet")

    # Tension proxy: silence and distress words up, chatter down. Dialogue
    # density is inverted because wall-to-wall talking reads as comedy or
    # drama, while tense sequences go quiet.
    density, silence, distress = arcs[:, 0], arcs[:, 1], arcs[:, 2]

    def z(x):
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True)
        return (x - mu) / np.maximum(sd, 1e-6)

    tension = z(silence) + z(distress) - 0.5 * z(density)
    # Light smoothing: deciles are noisy and the shape is what matters.
    k = np.array([0.25, 0.5, 0.25])
    tension = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, tension)
    shape = z(tension)

    km = KMeans(n_clusters=args.k, n_init=10, random_state=SEED).fit(shape)
    labels = km.labels_
    names, seen = [], {}
    for i, c in enumerate(km.cluster_centers_):
        nm = name_shape(c)
        seen[nm] = seen.get(nm, 0) + 1
        names.append(nm if seen[nm] == 1 else f"{nm} {seen[nm]}")

    print(f"{len(rows)} movies · {args.k} archetypes")
    for i, nm in enumerate(names):
        m = labels == i
        ex = movies["title"].iloc[[rows[j] for j in np.flatnonzero(m)[:4]]].tolist()
        curve = " ".join(f"{v:+.1f}" for v in km.cluster_centers_[i])
        print(f"  {nm:16} n={m.sum():4d}  [{curve}]")
        print(f"       e.g. {', '.join(ex)}")

    out = {
        "archetypes": names,
        "centroids": [[round(float(v), 3) for v in c] for c in km.cluster_centers_],
        "movies": {str(rows[i]): {"arc": [round(float(v), 2) for v in shape[i]],
                                  "type": int(labels[i])}
                   for i in range(len(rows))},
    }
    (DATA_DIR / "arcs.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"\nWrote {len(rows)} arcs -> data/arcs.json")


if __name__ == "__main__":
    main()
