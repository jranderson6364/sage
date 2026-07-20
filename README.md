# Sage

An interactive map of movies that doubles as a recommender.

Several thousand films are positioned in 2D space by similarity and rendered as an
explorable WebGL graph. Click a movie and its nearest neighbors light up — that's the
recommendation. Toggle between two notions of "similar":

- **Similar story** — plot/keyword text embeddings (content-based)
- **Similar audience** — real user ratings via ALS matrix factorization (collaborative filtering)

## How it works

Everything is precomputed offline in Python and exported as static JSON — no backend.

```
TMDB + MovieLens → embeddings / ALS → UMAP 2D layout → k-NN neighbors → static JSON → sigma.js
```

## Repo layout

- `pipeline/` — Python data + ML pipeline (fetch, embed, layout, export)
- `web/` — sigma.js front end (deployed to GitHub Pages)

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
python layout_umap.py                                 # 9. 2D map layout (UMAP)
python export_web.py                                  # 10. static JSON -> web/public/data/

python evaluate_lenses.py     # optional: score channels against co-rating truth
```

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
