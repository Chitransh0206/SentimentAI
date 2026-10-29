import pandas as pd
import pytest

from config import settings
from src.data_loader import (
    DataValidationError, InsufficientDataError, class_balance_warnings, clean_labeled_dataframe,
    load_dataset, normalize_label, ratings_to_sentiment, read_uploaded_csv, validate_class_sufficiency,
)


def write(tmp_path, df, name="d.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def test_valid_csv_loads(tmp_path):
    path = write(tmp_path, pd.DataFrame({"text": ["good", "bad"], "sentiment": ["Positive", "NEG"]}))
    df, report = load_dataset(path)
    assert list(df.columns) == ["text", "sentiment"]
    assert df["sentiment"].tolist() == ["positive", "negative"]
    assert report.rows_final == 2


def test_missing_file_message(tmp_path):
    with pytest.raises(DataValidationError, match="not found"):
        load_dataset(tmp_path / "nope.csv")


def test_empty_file(tmp_path):
    path = tmp_path / "e.csv"
    path.write_text("")
    with pytest.raises(DataValidationError, match="empty"):
        load_dataset(path)


def test_missing_columns_detected(tmp_path):
    path = write(tmp_path, pd.DataFrame({"review": ["x"], "label": ["positive"]}))
    with pytest.raises(DataValidationError, match="Missing required column"):
        load_dataset(path)


def test_column_mapping(tmp_path):
    path = write(tmp_path, pd.DataFrame({"review": ["x"], "label": ["neutral"]}))
    df, _ = load_dataset(path, text_column="review", label_column="label")
    assert df.loc[0, "sentiment"] == "neutral"


def test_invalid_labels_rejected_by_default():
    df = pd.DataFrame({"text": ["a", "b"], "sentiment": ["positive", "meh"]})
    with pytest.raises(DataValidationError, match="unsupported labels"):
        clean_labeled_dataframe(df)


def test_invalid_labels_dropped_and_reported_when_requested():
    df = pd.DataFrame({"text": ["a", "b"], "sentiment": ["positive", "meh"]})
    out, report = clean_labeled_dataframe(df, drop_unsupported_labels=True)
    assert len(out) == 1 and report.removed_unsupported_label == 1 and "meh" in report.unsupported_labels_seen


def test_missing_and_empty_text_removed():
    df = pd.DataFrame({"text": ["ok", "", "   ", None], "sentiment": ["positive"] * 4})
    out, report = clean_labeled_dataframe(df)
    assert len(out) == 1 and report.removed_missing_text == 3


def test_duplicates_and_conflicts():
    df = pd.DataFrame({"text": ["a", "a", "b", "b", "c"],
                       "sentiment": ["positive", "positive", "positive", "negative", "neutral"]})
    out, report = clean_labeled_dataframe(df)
    assert report.removed_exact_duplicates == 1
    assert report.removed_conflicting_duplicates == 2
    assert sorted(out["text"]) == ["a", "c"]


def test_numeric_labels_not_guessed():
    assert normalize_label("1") is None
    assert normalize_label("1", {"1": "positive"}) == "positive"


def test_ratings_conversion():
    out = ratings_to_sentiment(pd.Series(["1", "3", "5", "x"]))
    assert out.tolist()[:3] == ["negative", "neutral", "positive"] and pd.isna(out.iloc[3])
    with pytest.raises(DataValidationError):
        ratings_to_sentiment(pd.Series([1]), negative_max=4, positive_min=2)


def test_insufficient_data_error():
    with pytest.raises(InsufficientDataError, match="'neutral' has 0"):
        validate_class_sufficiency({"positive": 100, "negative": 100})


def test_imbalance_and_size_warnings():
    warnings = class_balance_warnings({"positive": 3000, "negative": 100, "neutral": 100})
    assert any("imbalance" in w for w in warnings)
    assert any(str(settings.RECOMMENDED_TOTAL_SAMPLES) in w for w in class_balance_warnings({"positive": 5, "negative": 5, "neutral": 5}))


def test_uploaded_csv_validation():
    assert len(read_uploaded_csv(b"text\nhello\nworld\n")) == 2
    for bad, msg in [(b"", "empty"), (b"text\n", "no data rows"), (b"\xff\xfe\x00bad", "UTF-8")]:
        with pytest.raises(DataValidationError, match=msg):
            read_uploaded_csv(bad)
    with pytest.raises(DataValidationError, match="limit"):
        read_uploaded_csv(b"text\n" + b"a\n" * 10, max_rows=5)
    with pytest.raises(DataValidationError, match="MB"):
        read_uploaded_csv(b"text\n" + b"a" * 2_000_000, max_mb=1)
