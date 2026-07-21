# Sage

An interactive map of movies that doubles as a recommender.

Several thousand films float in a 3D space whose axes are how the movie *feels*, not
what it's about:

- **Levity** — how heavily it takes itself (somber ↔ playful)
- **Threat** — tension, stakes, danger (safe ↔ tense)
- **Intimacy** — focus on close emotional connection (detached ↔ intimate)

Dot size is rating, hue blends the three moods. Click a movie and its recommendations
light up; everything else clears away. Pick tags describing what you liked about it and
the recommendations re-rank around that aspect.

## How it works

Everything is precomputed offline in Python and exported as static JSON — no backend.

```
TMDB + MovieLens → embeddings / ALS / tag genome / reviews
                 → semantic axes + fused k-NN → static JSON → three.js
```

Recommendations are one model: weighted reciprocal-rank fusion of three channels —
story embeddings, MovieLens tag genome, and ALS audience factors. The channels aren't
exposed separately; they exist to feed the fusion.

## Repo layout

- `pipeline/` — Python data + ML pipeline (fetch, embed, score, export)
- `web/` — three.js front end (deployed to GitHub Pages)

## Running the pipeline

Needs a TMDB API token in `.env` (`TMDB_READ_ACCESS_TOKEN=...`) and the
MovieLens zip unpacked into `pipeline/data/` (ml-latest-small for dev).

```
cd pipeline
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt

python fetch_tmdb.py                                  # 1. top-5000 movies from TMDB (cached, resumable)
python join_movielens.py --ml-dir data/ml-25m         # 2. attach MovieLens ids + export ratings
python embed_text.py                                  # 3. story embeddings (sentence-transformers)
python train_als.py --factors 128 --min-ratings 10    # 4. audience factors (implicit ALS)
python build_genome.py                                # 5. tag-genome "vibe" vectors
python fetch_reviews.py                               # 6. TMDB user reviews (cached, resumable)
python embed_reviews.py                               # 7. mean-pooled review embeddings
python semantic_axes.py                               # 8. levity/threat/intimacy axis scores
python export_web.py                                  # 9. static JSON -> web/public/data/

python evaluate_lenses.py     # optional: score recommendation channels vs co-rating truth
python evaluate_axes.py       # optional: score axes vs the hand-labeled validation set
```

`evaluate_axes.py` is the guard rail for axis changes: it reports Spearman
correlation against `axis_labels.json` (104 hand-scored movies), split by tag-genome
coverage, plus `sep` — the percentile gap between films labeled extreme-low and
extreme-high. A change that raises correlation while shrinking `sep` is flattening the
cloud toward the middle rather than improving it, so both need to move the right way.

## Running the web app

```
cd web
npm install
npm run dev                   # http://localhost:5173
```

## Status

Pipeline scripts complete end-to-end; front end MVP in progress.

## Attribution

This product uses the TMDB API but is not endorsed or certified by TMDB.
Ratings data from [MovieLens](https://grouplens.org/datasets/movielens/) (GroupLens Research).
