"""Re-embed reviews keeping only the sentences that describe how a film felt.

embed_reviews.py mean-pools whole reviews, which dilutes the useful part.
Most review text is about craft and legacy — cinematography, the director,
awards, how it holds up — and averaging that together with "the tension is
unbearable" washes the tone out. That's the diagnosed cause of Jurassic
Park's threat sitting at the 36th percentile while its genome tags correctly
said 68th: its reviews are about wonder and groundbreaking effects.

So: split each review into sentences, embed them, and keep only those closer
to a "how it felt to watch" probe than to a "how it was made" probe. Pool
what survives. Same output shape as embed_reviews.py, so it's a drop-in
swap for the review feature block.

Reads data/reviews.json, writes data/review_tone_emb.npy and
data/review_tone_rows.json.

Usage:
    python embed_reviews_tone.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent / "data"
MODEL = "all-MiniLM-L6-v2"

# Two probes. A sentence is kept when it's closer to the first than the
# second — i.e. it's describing the experience, not the production.
FEEL_PROBES = [
    "it was terrifying and I could not relax the whole time",
    "hilarious, I laughed out loud constantly",
    "it broke my heart, I cried",
    "deeply moving and intimate, I cared about these characters",
    "unbearably tense, my heart was pounding",
    "warm and comforting, a joy to watch",
]
CRAFT_PROBES = [
    "the cinematography and visual effects are remarkable",
    "the director's best work, brilliantly edited",
    "it won several Oscars and deserved them",
    "the screenplay is well structured and the pacing works",
    "a landmark film that influenced everything after it",
    "the lead actor gives a career best performance",
]

# Reviews are prose, so a naive split on sentence punctuation is adequate and
# avoids pulling in a sentence-tokenizer dependency.
SPLIT = re.compile(r"(?<=[.!?])\s+")
MIN_WORDS = 4


def main() -> None:
    movies = pd.read_parquet(DATA_DIR / "movies.parquet")
    reviews = json.loads((DATA_DIR / "reviews.json").read_text())

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL)
    feel = model.encode(FEEL_PROBES, normalize_embeddings=True).mean(axis=0)
    craft = model.encode(CRAFT_PROBES, normalize_embeddings=True).mean(axis=0)
    feel /= np.linalg.norm(feel)
    craft /= np.linalg.norm(craft)

    tmdb_to_row = {int(t): i for i, t in enumerate(movies["tmdb_id"])}
    rows, sentences, splits = [], [], [0]
    for tmdb_id, texts in reviews.items():
        row = tmdb_to_row.get(int(tmdb_id))
        if row is None or not texts:
            continue
        sents = []
        for t in texts:
            for s in SPLIT.split(t.replace("\r", " ").replace("\n", " ")):
                if len(s.split()) >= MIN_WORDS:
                    sents.append(s.strip())
        if not sents:
            continue
        rows.append(row)
        sentences.extend(sents)
        splits.append(len(sentences))

    print(f"{len(rows)} movies · {len(sentences)} sentences to embed")
    emb = model.encode(sentences, batch_size=256, show_progress_bar=True,
                       normalize_embeddings=True).astype(np.float32)

    keep_score = emb @ feel - emb @ craft
    pooled = np.zeros((len(rows), emb.shape[1]), dtype=np.float32)
    kept_total = 0
    for i in range(len(rows)):
        chunk = emb[splits[i]: splits[i + 1]]
        sc = keep_score[splits[i]: splits[i + 1]]
        keep = chunk[sc > 0]
        # If a film's reviews are *entirely* craft talk, fall back to the
        # whole review rather than emitting a zero vector.
        if len(keep) == 0:
            keep = chunk
        kept_total += len(keep)
        v = keep.mean(axis=0)
        pooled[i] = v / max(np.linalg.norm(v), 1e-12)

    np.save(DATA_DIR / "review_tone_emb.npy", pooled)
    (DATA_DIR / "review_tone_rows.json").write_text(json.dumps(rows))
    print(f"kept {kept_total}/{len(sentences)} sentences "
          f"({kept_total / len(sentences):.1%}) as tone-bearing")
    print(f"wrote {pooled.shape} -> data/review_tone_emb.npy")


if __name__ == "__main__":
    main()
