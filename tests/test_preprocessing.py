import math

import pytest

from src.preprocessing import is_blank, normalize_text
from src.train import make_vectorizer


def test_basic_normalization():
    assert normalize_text("  Great   PRODUCT!!  ") == "great product!!"


@pytest.mark.parametrize("value", ["", "   ", "\n\t", None, float("nan")])
def test_empty_and_missing_become_empty_string(value):
    assert normalize_text(value) == ""
    assert is_blank(value)


def test_non_string_raises():
    with pytest.raises(TypeError):
        normalize_text(123)


def test_unicode_and_curly_apostrophes():
    assert normalize_text("I don\u2019t like it") == "i don't like it"
    assert normalize_text("ｆｕｌｌｗｉｄｔｈ") == "fullwidth"


def test_url_html_and_repeats():
    assert normalize_text("see https://x.com/a <b>now</b>") == "see url now"
    assert normalize_text("soooooo good") == "sooo good"


def test_negation_is_preserved_in_features():
    analyzer = make_vectorizer().build_analyzer()
    tokens = analyzer("This is NOT good, never buying again")
    for kept in ("not", "never", "not good", "never buying"):
        assert kept in tokens


def test_exclamation_kept_as_token():
    assert "!" in make_vectorizer().build_analyzer()("great!")


def test_idempotent():
    once = normalize_text("  Mixed   CASE  ")
    assert normalize_text(once) == once
    assert not math.isnan(len(once))
