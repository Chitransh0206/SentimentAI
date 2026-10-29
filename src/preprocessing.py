"""Text normalisation shared by training and inference.

Design notes
------------
* ONE function (``normalize_text``) is used everywhere. It is plugged into
  ``TfidfVectorizer(preprocessor=...)`` so it runs inside the saved Pipeline and
  training/inference can never drift apart.
* We deliberately do NOT remove stop words, strip all punctuation, stem or
  lemmatize. Words such as "not", "no" and "never" carry sentiment, and
  ``!``/``?`` are kept as tokens. Bigrams ("not good") are produced by the
  vectorizer's ``ngram_range``.
* Only the standard library is used, so there are no tokenizer downloads that
  could be missing at run time.
"""
from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_REPEAT_CHAR_RE = re.compile(r"(.)\1{3,}")  # "soooooo" -> "sooo"
_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'", "`": "'"})

# Keeps contractions ("don't") and the sentiment-bearing punctuation "!" and "?".
TOKEN_PATTERN = r"(?u)\b\w[\w']*\b|[!?]"


def is_missing(value: Any) -> bool:
    """True for None and NaN-like values (pandas NA / float nan)."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:  # pandas.NA and NaT raise on bool(); treat them as missing
        import pandas as pd

        return bool(pd.isna(value)) if not isinstance(value, (str, bytes, list, tuple, dict)) else False
    except (TypeError, ValueError):
        return False


def normalize_text(value: Any) -> str:
    """Return a normalised string for ``value``.

    * ``None``/NaN -> ``""`` (missing text is handled safely, never raises).
    * ``str`` -> Unicode NFKC, curly apostrophes fixed, HTML tags and URLs
      replaced, runs of repeated characters shortened, lower-cased, whitespace
      collapsed.
    * Any other type -> ``TypeError`` (callers must convert explicitly).
    """
    if is_missing(value):
        return ""
    if not isinstance(value, str):
        raise TypeError(f"Expected text as str, got {type(value).__name__}.")
    text = unicodedata.normalize("NFKC", value).translate(_APOSTROPHES)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" url ", text)
    text = _REPEAT_CHAR_RE.sub(r"\1\1\1", text)
    text = text.lower()
    return _WHITESPACE_RE.sub(" ", text).strip()


def is_blank(value: Any) -> bool:
    """True when ``value`` is missing or contains only whitespace."""
    if is_missing(value):
        return True
    return isinstance(value, str) and not value.strip()
