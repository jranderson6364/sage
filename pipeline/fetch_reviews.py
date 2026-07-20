"""Fetch TMDB user reviews per movie for the semantic-axis "review" signal.

Reviews name tone explicitly ("hilarious", "terrifying", "made me cry") in
ways a two-sentence overview rarely does. Same cache-one-file-per-movie
pattern as fetch_tmdb.py, so an interrupted fetch just resumes; an empty
cache file means "checked, no reviews" (not "not yet fetched").

Reads pipeline/data/movies.parquet, writes one JSON file per movie under
pipeline/data/raw_reviews/, and a combined pipeline/data/reviews.json
mapping tmdb_id (str) -> list of review text (only movies with >=1 review).

Usage:
    python fetch_reviews.py
    python fetch_reviews.py --max-pages 3   # up to 60 reviews/movie (default 20)
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
RAW_DIR = DATA_DIR / "raw_reviews"

API_BASE = "https://api.themoviedb.org/3"
REQUEST_DELAY_S = 0.15

# The embedding model truncates around ~256 tokens anyway; this just bounds
# worst-case memory for the handful of review-as-essay outliers.
MAX_CHARS_PER_REVIEW = 3000


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


def fetch_reviews(session: requests.Session, tmdb_id: int, max_pages: int) -> list[str]:
    """Review text for one movie, cached as one JSON file (20 reviews/page)."""
    cache = RAW_DIR / f"{tmdb_id}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    texts: list[str] = []
    for page in range(1, max_pages + 1):
        data = get(session, f"/movie/{tmdb_id}/reviews", page=page, language="en-US")
        texts.extend(
            r["content"][:MAX_CHARS_PER_REVIEW]
            for r in data.get("results", [])
            if r.get("content")
        )
        if page >= data.get("total_pages", 0):
            break

    cache.write_text(json.dumps(texts), encoding="utf-8")
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-pages", type=int, default=1,
        help="TMDB review pages per movie, 20 reviews/page (default 1)",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()

    movies = pd.read_parquet(DATA_DIR / "movies.parquet")
    reviews: dict[str, list[str]] = {}
    for tmdb_id in tqdm(movies["tmdb_id"], desc="reviews"):
        texts = fetch_reviews(session, int(tmdb_id), args.max_pages)
        if texts:
            reviews[str(int(tmdb_id))] = texts

    out = DATA_DIR / "reviews.json"
    out.write_text(json.dumps(reviews), encoding="utf-8")
    covered = len(reviews)
    total_reviews = sum(len(v) for v in reviews.values())
    print(
        f"{covered}/{len(movies)} movies have reviews "
        f"({covered / len(movies):.1%} coverage), {total_reviews} reviews total"
    )


if __name__ == "__main__":
    main()
