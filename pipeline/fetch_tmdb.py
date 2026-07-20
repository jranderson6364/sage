"""Fetch top-N movies from TMDB and cache them locally.

Two stages, both idempotent:
  1. Discover movie IDs by vote count (proxy for "popular + enough ratings to matter").
  2. Fetch full details (+ keywords, credits) per movie, one cached JSON file each.
Then flatten everything into pipeline/data/movies.parquet.

Re-running skips anything already cached, so an interrupted fetch just resumes.

Usage:
    python fetch_tmdb.py            # default: top 5000 movies
    python fetch_tmdb.py --n 8000
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

import os

PIPELINE_DIR = Path(__file__).parent
REPO_ROOT = PIPELINE_DIR.parent
DATA_DIR = PIPELINE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

API_BASE = "https://api.themoviedb.org/3"

# TMDB allows ~50 req/s but asks for courtesy; this keeps us far under any limit
# while still fetching 5k movies in ~15 minutes.
REQUEST_DELAY_S = 0.15

# Discover results are paged 20/page, hard-capped at page 500 (10k movies max).
PAGE_SIZE = 20
MAX_PAGE = 500


def make_session() -> requests.Session:
    load_dotenv(REPO_ROOT / ".env")
    token = os.getenv("TMDB_READ_ACCESS_TOKEN")
    session = requests.Session()
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    else:
        key = os.getenv("TMDB_API_KEY")
        if not key:
            raise SystemExit(
                "No TMDB credentials found. Put TMDB_READ_ACCESS_TOKEN (preferred) "
                f"or TMDB_API_KEY in {REPO_ROOT / '.env'}"
            )
        session.params = {"api_key": key}
    return session


def get(session: requests.Session, path: str, **params) -> dict:
    for attempt in range(5):
        resp = session.get(f"{API_BASE}{path}", params=params, timeout=30)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 2))
            time.sleep(wait + 1)
            continue
        resp.raise_for_status()
        time.sleep(REQUEST_DELAY_S)
        return resp.json()
    raise RuntimeError(f"Giving up on {path} after repeated 429s")


def discover_ids(session: requests.Session, n: int) -> list[int]:
    """Top-n movie IDs by vote count. Cached as one JSON list."""
    cache = DATA_DIR / f"discover_ids_{n}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    ids: list[int] = []
    n_pages = min((n + PAGE_SIZE - 1) // PAGE_SIZE, MAX_PAGE)
    for page in tqdm(range(1, n_pages + 1), desc="discover"):
        data = get(
            session,
            "/discover/movie",
            sort_by="vote_count.desc",
            page=page,
            include_adult="false",
        )
        ids.extend(m["id"] for m in data["results"])
        if page >= data["total_pages"]:
            break

    ids = ids[:n]
    cache.write_text(json.dumps(ids))
    return ids


def fetch_movie(session: requests.Session, movie_id: int) -> dict:
    """Full movie details with keywords + credits, cached one file per movie."""
    cache = RAW_DIR / f"{movie_id}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    data = get(session, f"/movie/{movie_id}", append_to_response="keywords,credits")
    cache.write_text(json.dumps(data), encoding="utf-8")
    return data


def flatten(movie: dict) -> dict:
    """One tidy row per movie for the parquet export."""
    directors = [
        c["name"] for c in movie.get("credits", {}).get("crew", [])
        if c.get("job") == "Director"
    ]
    cast = [c["name"] for c in movie.get("credits", {}).get("cast", [])[:8]]
    return {
        "tmdb_id": movie["id"],
        "imdb_id": movie.get("imdb_id"),
        "title": movie.get("title"),
        "year": (movie.get("release_date") or "")[:4] or None,
        "genres": [g["name"] for g in movie.get("genres", [])],
        "overview": movie.get("overview"),
        "keywords": [k["name"] for k in movie.get("keywords", {}).get("keywords", [])],
        "tagline": movie.get("tagline"),
        "directors": directors,
        "cast": cast,
        "poster_path": movie.get("poster_path"),
        "popularity": movie.get("popularity"),
        "vote_count": movie.get("vote_count"),
        "vote_average": movie.get("vote_average"),
        "runtime": movie.get("runtime"),
        "original_language": movie.get("original_language"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5000, help="number of movies to fetch")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()

    ids = discover_ids(session, args.n)
    # Discover pagination isn't stable — the same movie can appear on two
    # pages, so the cached list may hold duplicates. Dedupe, preserving order.
    ids = list(dict.fromkeys(ids))
    print(f"{len(ids)} unique movie IDs discovered")

    rows = []
    for movie_id in tqdm(ids, desc="movies"):
        rows.append(flatten(fetch_movie(session, movie_id)))

    df = pd.DataFrame(rows)
    out = DATA_DIR / "movies.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df)} movies -> {out}")
    print(f"With overview: {df['overview'].str.len().gt(0).sum()}, "
          f"with keywords: {(df['keywords'].str.len() > 0).sum()}, "
          f"with poster: {df['poster_path'].notna().sum()}")


if __name__ == "__main__":
    main()
