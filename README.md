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
python semantic_axes.py                               # 8. hand-tuned axis scores (fallback/baseline)
python train_axes.py                                  # 9. learned axis scores (what ships)
python fetch_subtitles.py --index                     # 10. index the remote 25GB subtitle archive
python fetch_subtitles.py --fetch                     #     range-fetch only our ~3.7k films (~390MB)
python subtitle_features.py                           # 11. timing features + per-decile curves
python narrative_arcs.py                              # 12. cluster tension shapes into archetypes
python fetch_lists.py                                 # 13. resolve curated lists against the catalogue
python export_web.py                                  # 14. static JSON -> web/public/data/

python evaluate_lenses.py     # optional: score recommendation channels vs co-rating truth
python evaluate_axes.py       # optional: score axes vs the hand-labeled validation set
```

### How the axes are scored

Two implementations, both kept:

- `semantic_axes.py` — hand-built. Sums curated tag groups, projects anchor phrases,
  blends three channels with per-axis weights. Interpretable, no labels required.
- `train_axes.py` — **fitted, and what ships.** Ridge regression over all 1,128 tag
  dimensions plus story and review embeddings, trained on `axis_labels_train.json`
  (279 hand-scored movies) and reported on the held-out `axis_labels.json` (104 more).
  Learns feature selection and channel weighting instead of guessing them.

Because it regresses onto the 1–10 label scale, its output is a predicted human rating
rather than a forced-uniform percentile — so the axis distributions come out shaped like
reality (most films aren't comedies) instead of flat by construction.

`evaluate_axes.py` is the guard rail for any axis change: Spearman correlation on the
held-out labels, split by tag-genome coverage, plus `sep` — the percentile gap between
films labeled extreme-low and extreme-high. A change that raises correlation while
shrinking `sep` is flattening the cloud toward the middle rather than improving it, so
both have to move the right way.

Held-out test, hand-tuned → learned: levity .825 → **.855**, threat .809 → **.876**,
intimacy .691 → **.756**.

### Narrative arcs

Every other signal here reduces a film to a point. Subtitle *timing* gives something
tags and plot summaries can't: how a film moves. A thriller that ratchets steadily and
one that explodes in the last act score identically on the threat axis and feel nothing
alike.

`fetch_subtitles.py` reads the OPUS OpenSubtitles corpus — a single 25GB ZIP64 archive
with no per-movie objects — over HTTP range requests: locate the central directory, pull
just that (~70MB), then fetch only the members whose path carries an IMDb id we have.
390MB transferred instead of 25GB, joined exactly on `imdb_id`.

`narrative_arcs.py` builds a tension curve per decile of runtime (silence and distress
vocabulary up, chatter down), z-normalizes so clustering keys on *shape* rather than
loudness, and k-means clusters it into six archetypes — Slow burn, Third-act climax,
Twin peaks, Cold open, Double climax, Midpoint + finale. Shown as a sparkline in the
detail panel. Related prior work: Reagan et al. found six shapes in books; a later study
reported the same on ~6k movie scripts. This does it on subtitle timing, which covers
far more films than scripts do.

The 18 scalar subtitle features were also tested as axis inputs and **measured not to
help** (+0.002–0.005 CV, −0.001–0.005 on held-out test — selection noise), so they're
off by default behind `train_axes.py --with-subs`. The arcs are a separate product
feature and don't depend on that result.

### Lists

Picking a list ("Best Picture Winners", "Best Romance") scopes the entire app to
its films — the cloud, the search box, and the recommendations all narrow
together, and the three mood filters keep working inside that smaller world.

The recommendations are the part that can't be done in the browser. A ~100-film
list is about 2% of the catalogue, so filtering a movie's global top-10
neighbours down to list members leaves roughly nothing. Instead `export_web.py`
re-ranks *every channel* over list members only and fuses them with the same
weights, so a list is the same recommender looking at a smaller world rather
than a different one. Selecting Eternal Sunshine of the Spotless Mind under
"Best Romance" suggests Amélie and Before Sunset; the same film under "Mind
Benders" suggests Memento and Donnie Darko.

Lists are defined in `pipeline/lists.json`, either as a public TMDB list id or
as hand-picked `[title, year]` pairs (written that way so the set is auditable
in review and a typo fails loudly instead of silently resolving to the wrong
film). `fetch_lists.py` resolves them and reports coverage, which is the number
that decides what ships: the catalogue is TMDB's top ~5000 by vote count, so
mainstream lists land near 100% while arthouse-leaning ones barely register
(measured: Golden Lion 11%, Palme d'Or 30%). Anything under `--min-coverage` is
dropped rather than shipped as a misleading stub, and a list that made the cut
but isn't complete says so in the menu ("70 of 99 in catalog").

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
