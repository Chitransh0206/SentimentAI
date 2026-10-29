"""Synthetic review generator used ONLY for demos and software tests.

The sentences are produced from templates, so any model trained on them will score
almost perfectly. That says nothing about real-world accuracy.
"""
from __future__ import annotations

import random

import pandas as pd

_SUBJECTS = ["The product", "The service", "This hotel", "The delivery", "The app", "The restaurant",
             "Our waiter", "The phone", "This laptop", "The support team"]
_POSITIVE = ["is excellent", "is absolutely wonderful", "works great and I love it", "is really good",
             "exceeded my expectations", "is fantastic value", "made me very happy", "is not bad at all, quite impressive",
             "is amazing and I would buy again", "is perfect"]
_NEGATIVE = ["is terrible", "is really disappointing", "is not good", "is awful and I am not happy",
             "broke after two days", "was a waste of money", "is the worst I have used",
             "is poor and I would never buy again", "is frustrating and unreliable", "is not worth it"]
_NEUTRAL = ["is okay, nothing special", "is average", "arrived on Tuesday", "is what the description says",
            "is fine, neither good nor bad", "does the job, no more and no less", "is acceptable",
            "is as expected", "comes in a standard box", "is ordinary"]
_CLOSERS = ["", " Overall.", " That is my view.", " Just saying.", " Honestly.", " Thanks."]


def make_synthetic_reviews(n_per_class: int = 150, seed: int = 42) -> pd.DataFrame:
    """Return a shuffled ``text,sentiment`` frame with unique rows."""
    rng = random.Random(seed)
    rows: list[tuple[str, str]] = []
    for label, phrases in (("positive", _POSITIVE), ("negative", _NEGATIVE), ("neutral", _NEUTRAL)):
        combos = [f"{s} {p}.{c}" for s in _SUBJECTS for p in phrases for c in _CLOSERS]
        rng.shuffle(combos)
        rows += [(text, label) for text in combos[:n_per_class]]
    rng.shuffle(rows)
    return pd.DataFrame(rows, columns=["text", "sentiment"])
