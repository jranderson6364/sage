"""Score every movie on interpretable semantic axes for the 3D view.

The three axes (definitions by the project owner):

  levity    How heavily the movie takes itself. Low = gritty, tragic,
            intensely serious (The Dark Knight). High = pure comedy, satire,
            absurdity (Superbad).
  threat    Tension, stakes, and danger — not just horror. Low = pure
            comfort, zero anxiety (Paddington). Medium = thrilling suspense
            (Inception). High = dread, terror, visceral panic (A Quiet Place).
  intimacy  Focus on close emotional human connection — romance, but equally
            brotherhood, friendship, family. Low = plot-driven, isolated
            characters (Mad Max: Fury Road). High = soulmates, deep bonds
            (The Notebook).

Three signals per axis, blended:
  - genome tags (curated tag groups per pole) — precise but only covers
    movies in the MovieLens tag genome.
  - reviews (mean-pooled, per-review anchor projection) — user reviews name
    tone explicitly ("hilarious", "terrifying") in ways overviews rarely do,
    but TMDB review coverage is sparse.
  - anchor-phrase projection of the story (overview) embedding — noisier,
    covers every movie, so it's the fallback when the others are absent.
    Text embeddings alone misread e.g. The Dark Knight as comedic ("Joker",
    "comic"), which is why genome/reviews outrank it whenever available.
All three are rank-normalized within their own coverage set, then blended
per movie using only the channels that movie actually has, weighted by
GENOME_WEIGHT/REVIEW_WEIGHT/TEXT_WEIGHT and renormalized to sum to 1. The
blend is re-ranked one more time to a final percentile in [-1, 1].

Reads data/text_emb.npy, data/genome.npy (+ genome_rows.json from
build_genome.py), data/review_emb.npy (+ review_rows.json from
embed_reviews.py). Writes data/axes.npy (float32, n x 3, columns in AXES
order), prints per-axis extremes, a signal-disagreement report (movies
where genome/review/text point different directions — worth an eyeball,
not necessarily wrong), and the CHECKS calibration list.

Usage:
    python semantic_axes.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Movie titles carry glyphs (fractions, accents, CJK) that the default
# Windows console codepage can't encode — without this the run dies partway
# through on whichever title happens to surface in a report.
sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent / "data"
MODEL = "all-MiniLM-L6-v2"  # must match embed_text.py / embed_reviews.py

AXES = {
    "levity": (
        ["a pure comedy designed entirely for laughs",
         "silly, absurd, playful fun that never takes itself seriously",
         "a lighthearted satirical romp full of jokes"],
        ["a grave, somber film that takes itself completely seriously",
         "gritty, bleak, and tragic with no humor at all",
         "an intensely serious, heavy drama about suffering"],
    ),
    "threat": (
        ["characters in constant danger, dread, and mortal peril",
         "heart-pounding terror, panic, and imminent threat",
         "relentless suspense and life-or-death stakes"],
        ["a cozy, comforting story where nothing bad happens",
         "completely safe, warm, and free of any danger",
         "gentle and soothing, with zero tension or anxiety"],
    ),
    "intimacy": (
        ["a story built on deep emotional bonds between people",
         "intense love, devoted friendship, and family at its heart",
         "characters who profoundly connect, trust, and care for each other"],
        ["isolated, detached characters with no close relationships",
         "a cold, plot-driven film focused on events rather than people",
         "lone survival with no emotional connection to anyone"],
    ),
}

# Genome tag groups per axis pole (curated against genome-tags.csv; a name
# that stops existing is silently skipped, so tweaks are safe).
GENOME_TAGS = {
    "levity": (
        ["comedy", "funny", "very funny", "funny as hell", "hilarious",
         "humor", "humorous", "goofy", "silly", "silly fun", "parody",
         "satire", "satirical", "witty", "quirky", "british comedy",
         "screwball comedy", "off-beat comedy", "crude humor", "sex comedy",
         "dumb but funny", "whimsical", "feel-good", "feel good movie"],
        ["dark", "bleak", "grim", "depressing", "depression", "tragedy",
         "gritty", "brutal", "brutality", "disturbing", "dark hero"],
    ),
    "threat": (
        ["tense", "suspense", "suspenseful", "scary", "horror", "creepy",
         "frightening", "intense", "violence", "violent",
         "gratuitous violence", "brutal", "brutality", "disturbing"],
        ["feel-good", "feel good movie", "heartwarming", "sweet",
         "whimsical", "kids and family"],
    ),
    "intimacy": (
        ["romance", "romantic", "love", "love story", "love triangles",
         "interracial romance", "romantic comedy", "good romantic comedies",
         "friendship", "unlikely friendships", "family", "family bonds",
         "family drama", "father son relationship", "father-son relationship",
         "father daughter relationship", "mother daughter relationship",
         "mother-son relationship", "relationships", "emotional",
         "sentimental", "touching", "heartwarming"],
        [],
    ),
}

# The owner's calibration examples plus a broader spread across all three
# axes — printed after scoring so anchor/weight tweaks can be judged against
# intent immediately, and so a change (like adding the review signal) can be
# diffed against a wider sample than the original 7 movies.
CHECKS = [
    "Paddington", "Inception", "A Quiet Place", "The Dark Knight", "Superbad",
    "Mad Max: Fury Road", "The Notebook",
    "Anchorman: The Legend of Ron Burgundy", "Airplane!",
    "Requiem for a Dream", "Schindler's List", "Hereditary",
    "The Grand Budapest Hotel", "Little Women", "Call Me by Your Name",
    "Before Sunrise", "John Wick", "Gravity", "The Godfather", "Toy Story",
    "Get Out", "Amelie",
]

# Channel weights, per axis, as (genome, review, text). Renormalized per
# movie over whichever channels it actually has.
#
# These differ by axis on purpose, and the split is mechanical rather than
# fitted noise: reviews and plot summaries describe comedy accurately
# (people write "hilarious"), so levity can lean on them — but for tension
# and relationships they describe craft, legacy and premise instead. The
# diagnostic case is Jurassic Park, whose genome tags correctly fire
# tense/suspense/scary while its reviews (wonder, nostalgia, effects) and
# its overview (a theme park) read as playful and safe, dragging threat
# down to the 36th percentile.
#
# Leaning this hard on genome only became safe once imputation gave every
# movie a tag profile; before that, a heavy genome weight would have
# stranded the ~20% with no tags. Sweeps: see evaluate_axes.py.
CHANNEL_WEIGHTS = {
    "levity": (0.50, 0.28, 0.22),
    "threat": (0.75, 0.14, 0.11),
    "intimacy": (0.78, 0.12, 0.10),
}


def rank01(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return (r + 0.5) / len(x)


def pct(v: float) -> int:
    return int(round((v + 1) / 2 * 100))


def impute_genome(emb, genome, genome_rows, n, k, power):
    """Borrow a tag profile for movies the tag genome doesn't cover.

    A movie with no community tags falls back to text+review anchors alone,
    which is measurably the weakest configuration (see evaluate_axes.py).
    Here each uncovered movie takes a similarity-weighted blend of its k
    nearest *covered* movies in story-embedding space.

    The obvious hazard is regression to the mean: averaging neighbors yields
    a blander profile than any real one, which would quietly pull genuinely
    extreme films toward the middle. `power` sharpens the weights so the
    closest neighbor dominates instead of the blend smearing across all k --
    watch evaluate_axes.py's `sep` column, not just rho, when tuning it.

    Returns (genome_all, imputed_mask).
    """
    covered = np.asarray(genome_rows)
    is_cov = np.zeros(n, dtype=bool)
    is_cov[covered] = True
    uncovered = np.flatnonzero(~is_cov)

    genome_all = np.zeros((n, genome.shape[1]), dtype=np.float32)
    genome_all[covered] = genome

    sim = emb[uncovered] @ emb[covered].T
    top = np.argpartition(-sim, k, axis=1)[:, :k]
    rows = np.arange(len(uncovered))[:, None]
    w = np.clip(sim[rows, top], 0, None) ** power
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    genome_all[uncovered] = np.einsum("ik,ikd->id", w, genome[top])

    return genome_all, ~is_cov


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Imputation is on by default: it lifts uncovered-movie accuracy hugely
    # (threat rho 0.394 -> 0.748) without flattening the extremes. --no-impute
    # reproduces the pre-imputation scoring for comparison.
    parser.add_argument("--no-impute", dest="impute", action="store_false",
                        help="don't give genome-uncovered movies a borrowed tag profile")
    parser.add_argument("--impute-k", type=int, default=10)
    parser.add_argument("--impute-power", type=float, default=5.0,
                        help="sharpen neighbor weights; higher = less averaging")
    parser.add_argument("--impute-weight", type=float, default=1.0,
                        help="scale the genome weight for imputed rows (0-1)")
    # Experiment overrides: if given, these replace CHANNEL_WEIGHTS for
    # every axis, so a sweep can test one global setting.
    parser.add_argument("--gw", type=float, help="override genome weight (all axes)")
    parser.add_argument("--rw", type=float, help="override review weight (all axes)")
    parser.add_argument("--tw", type=float, help="override text weight (all axes)")
    parser.add_argument("--out", default=str(DATA_DIR / "axes.npy"))
    args = parser.parse_args()

    emb = np.load(DATA_DIR / "text_emb.npy")
    movies = pd.read_parquet(DATA_DIR / "movies.parquet")
    n = len(emb)

    genome = np.load(DATA_DIR / "genome.npy")
    genome_rows = json.loads((DATA_DIR / "genome_rows.json").read_text())
    tag_names = json.loads((DATA_DIR / "genome_tags.json").read_text())
    tag_idx = {t: i for i, t in enumerate(tag_names)}

    review_emb = np.load(DATA_DIR / "review_emb.npy")
    review_rows = json.loads((DATA_DIR / "review_rows.json").read_text())
    genome_pos = {row: i for i, row in enumerate(genome_rows)}
    review_pos = {row: i for i, row in enumerate(review_rows)}
    print(f"genome covers {len(genome_rows)}/{n} movies, "
          f"reviews cover {len(review_rows)}/{n} movies, "
          f"overlap {len(set(genome_rows) & set(review_rows))}")

    # Which movies have a tag profile at all (real or borrowed).
    has_g = np.zeros(n, dtype=bool)
    has_g[genome_rows] = True
    if args.impute:
        genome_all, imputed = impute_genome(
            emb, genome, genome_rows, n, args.impute_k, args.impute_power)
        has_g |= imputed
        print(f"imputed tag profiles for {imputed.sum()} movies "
              f"(k={args.impute_k}, power={args.impute_power}, "
              f"weight x{args.impute_weight})")
    else:
        genome_all = np.zeros((n, genome.shape[1]), dtype=np.float32)
        genome_all[genome_rows] = genome
        imputed = np.zeros(n, dtype=bool)
    print()

    from sentence_transformers import SentenceTransformer  # slow import

    model = SentenceTransformer(MODEL)

    scores = np.empty((n, len(AXES)), dtype=np.float32)
    for col, (name, (pos, neg)) in enumerate(AXES.items()):
        gw, rw, tw = CHANNEL_WEIGHTS[name]
        if args.gw is not None: gw = args.gw
        if args.rw is not None: rw = args.rw
        if args.tw is not None: tw = args.tw
        # Imputed profiles are borrowed, not observed, so they can be
        # discounted relative to real ones via --impute-weight.
        gweight = np.zeros(n)
        gweight[has_g] = gw
        gweight[imputed] = gw * args.impute_weight
        anchors = model.encode(pos + neg, normalize_embeddings=True)
        direction = anchors[: len(pos)].mean(axis=0) - anchors[len(pos):].mean(axis=0)
        direction /= np.linalg.norm(direction)

        text_rank = rank01(emb @ direction)

        g_pos, g_neg = GENOME_TAGS[name]
        pos_i = [tag_idx[t] for t in g_pos if t in tag_idx]
        neg_i = [tag_idx[t] for t in g_neg if t in tag_idx]
        g_raw = genome_all[has_g][:, pos_i].sum(axis=1)
        if neg_i:
            g_raw = g_raw - genome_all[has_g][:, neg_i].sum(axis=1)
        # Ranked within the set that actually has a profile, then scattered
        # back — movies with no profile must not be ranked as "zero tags".
        g_rank = np.zeros(n)
        g_rank[has_g] = rank01(g_raw)

        review_rank = rank01(review_emb @ direction)

        # Weighted sum over whichever channels each movie actually has,
        # renormalized by the weight actually used (so e.g. a movie with
        # only text still lands on a plain 0-1 rank, not a fraction of one).
        weighted_sum = tw * text_rank + gweight * g_rank
        weight_used = tw + gweight
        weighted_sum[review_rows] += rw * review_rank
        weight_used[review_rows] += rw
        blended = weighted_sum / weight_used

        scores[:, col] = rank01(blended) * 2 - 1

        order = np.argsort(scores[:, col])
        top = movies["title"].iloc[order[-5:][::-1]].tolist()
        bottom = movies["title"].iloc[order[:5]].tolist()
        print(f"[{name}: {len(pos_i)}+{len(neg_i)} genome tags matched, "
              f"weights g{gw}/r{rw}/t{tw}]")
        print(f"most {name}:  {', '.join(top)}")
        print(f"least {name}: {', '.join(bottom)}")

        # Oddity triage: movies where genome/review/text disagree sharply on
        # this axis are worth an eyeball — could be a genuinely mixed-tone
        # film, could be one signal misreading it. Spread, not error.
        overlap = sorted(set(genome_rows) & set(review_rows))
        if overlap:
            spread = [
                # g_rank is full-length and scattered; review_rank is still
                # indexed by position within the review subset.
                (row, g_rank[row], review_rank[review_pos[row]], text_rank[row])
                for row in overlap
            ]
            spread.sort(key=lambda t: -(max(t[1:]) - min(t[1:])))
            print(f"most-disagreeing on {name} (genome / review / text):")
            for row, g, r, t in spread[:5]:
                title = movies["title"].iloc[row]
                print(f"  {title:40} {g*100:3.0f} / {r*100:3.0f} / {t*100:3.0f}")
        print()

    print("calibration checks (percentiles):")
    for title in CHECKS:
        hits = movies.index[movies["title"] == title]
        if len(hits) == 0:
            print(f"  {title:40} (not in catalog)")
            continue
        # A title can match multiple releases (e.g. Little Women 1994/2019) —
        # print every match instead of silently grabbing hits[0], since that
        # movie's own version can otherwise go unchecked.
        for i in hits:
            label = f"{title} ({movies['year'].iloc[i]})" if len(hits) > 1 else title
            print(f"  {label:40} levity {pct(scores[i, 0]):3d} · "
                  f"threat {pct(scores[i, 1]):3d} · intimacy {pct(scores[i, 2]):3d}")

    np.save(args.out, scores)
    print(f"\nWrote {scores.shape} axis scores -> {args.out}")


if __name__ == "__main__":
    main()
