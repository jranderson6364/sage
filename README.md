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

## Status

Early days — pipeline under construction.

## Attribution

This product uses the TMDB API but is not endorsed or certified by TMDB.
Ratings data from [MovieLens](https://grouplens.org/datasets/movielens/) (GroupLens Research).
