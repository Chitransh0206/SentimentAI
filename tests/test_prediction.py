import numpy as np
import pandas as pd
import pytest

from config import settings
from src.data_loader import DataValidationError
from src.model_utils import ModelNotFoundError, load_pipeline, save_artifacts
from src.predict import InvalidInputError, SentimentPredictor, predict_dataframe, to_csv_bytes
from src.train import split_data, train_and_evaluate
from src.evaluate import compute_metrics


def test_split_has_no_overlap_and_is_stratified(synthetic_df):
    train, test = split_data(synthetic_df, 0.2, 3)
    assert set(train.index).isdisjoint(test.index)
    assert len(train) + len(test) == len(synthetic_df)
    assert test["sentiment"].value_counts().nunique() == 1  # equal classes -> equal test counts


def test_training_is_deterministic(synthetic_df):
    a = train_and_evaluate(synthetic_df, seed=5)
    b = train_and_evaluate(synthetic_df, seed=5)
    assert a.metrics["validation"] == b.metrics["validation"]
    assert a.test_index == b.test_index


def test_selection_uses_validation_metric(training_result):
    rows = training_result.metrics["validation"]
    best = max(rows, key=lambda r: r["cv_f1_macro_mean"])
    assert training_result.metrics["selected_model"] == best["model"]
    assert len(rows) == 3


def test_metrics_structure(training_result):
    t = training_result.metrics["test"]
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted",
                "per_class", "confusion_matrix", "classification_report"):
        assert key in t
    assert list(t["per_class"]) == list(settings.LABELS)
    assert training_result.metadata["n_train"] + training_result.metadata["n_test"] == 180


def test_metrics_zero_division_for_unpredicted_class():
    m = compute_metrics(["positive", "negative", "neutral"], ["positive", "positive", "positive"])
    assert m["per_class"]["negative"]["precision"] == 0.0


def test_save_and_load_roundtrip(training_result, artifact_root):
    paths = save_artifacts(training_result.pipeline, training_result.metrics, training_result.metadata)
    assert paths.model.is_file() and paths.metrics.is_file() and paths.metadata.is_file()
    loaded = load_pipeline(paths.model)
    texts = ["This product is excellent.", "The service is terrible."]
    assert list(loaded.predict(texts)) == list(training_result.pipeline.predict(texts))
    assert list(loaded.classes_) == list(settings.LABELS)


def test_missing_model_gives_instructions(artifact_root):
    with pytest.raises(ModelNotFoundError, match="train_model.py"):
        load_pipeline()


def test_corrupt_model_file(artifact_root):
    path = settings.get_artifact_paths().model
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a model")
    with pytest.raises(ModelNotFoundError, match="could not be loaded"):
        load_pipeline()


def test_single_prediction(predictor):
    result = predictor.predict_one("The product is excellent and I love it.")
    assert result.label == "positive"
    assert result.score_kind == "probability"
    assert set(result.scores) == set(settings.LABELS)
    assert sum(result.scores.values()) == pytest.approx(1.0)
    assert predictor.predict_one("The service is not good.").label == "negative"


@pytest.mark.parametrize("bad", ["", "   ", None, 42, float("nan"), "x" * (settings.MAX_TEXT_CHARS + 1)])
def test_invalid_single_inputs(predictor, bad):
    with pytest.raises(InvalidInputError):
        predictor.predict_one(bad)


def test_batch_predictions_preserve_rows_and_report_skips(predictor):
    df = pd.DataFrame({"id": ["1", "2", "3", "4"], "review": ["The product is excellent.", None, "  ", "The service is awful and not good."]})
    out, summary = predict_dataframe(predictor, df, "review", chunk_size=2)
    assert len(out) == 4 and list(out["id"]) == ["1", "2", "3", "4"]
    assert out["predicted_sentiment"].tolist()[0] == "positive"
    assert pd.isna(out["predicted_sentiment"].iloc[1])
    assert summary.total_rows == 4 and summary.valid_rows == 2 and summary.skipped_rows == 2
    assert summary.skipped_reasons == {"skipped: empty text": 2}
    assert sum(summary.counts.values()) == 2
    assert {"probability_negative", "probability_neutral", "probability_positive"} <= set(out.columns)
    assert out.loc[[1, 2], "probability_positive"].isna().all()


def test_batch_matches_single_predictions(predictor, synthetic_df):
    sample = synthetic_df.head(25)
    out, _ = predict_dataframe(predictor, sample, "text", chunk_size=10)
    singles = [predictor.predict_one(t).label for t in sample["text"]]
    assert out["predicted_sentiment"].tolist() == singles


def test_batch_missing_column_and_empty_frame(predictor):
    with pytest.raises(DataValidationError, match="not found"):
        predict_dataframe(predictor, pd.DataFrame({"a": ["x"]}), "text")
    with pytest.raises(DataValidationError, match="no rows"):
        predict_dataframe(predictor, pd.DataFrame({"text": []}), "text")


def test_batch_column_name_collision(predictor):
    df = pd.DataFrame({"text": ["The product is excellent."], "predicted_sentiment": ["orig"]})
    out, _ = predict_dataframe(predictor, df, "text")
    assert out["predicted_sentiment"].iloc[0] == "orig"
    assert out["sentimentai_predicted_sentiment"].iloc[0] == "positive"


def test_csv_export_roundtrip_utf8(predictor):
    df = pd.DataFrame({"text": ["The product is excellent. café ☕"]})
    out, _ = predict_dataframe(predictor, df, "text")
    data = to_csv_bytes(out)
    assert data.startswith(b"\xef\xbb\xbf")
    back = pd.read_csv(pd.io.common.BytesIO(data), encoding="utf-8-sig")
    assert back["text"].iloc[0] == df["text"].iloc[0]
    assert back["predicted_sentiment"].iloc[0] == "positive"


def test_decision_score_fallback():
    """A pipeline without predict_proba must expose scores as decision scores, not probabilities."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    texts = ["good great", "bad awful", "okay fine"] * 5
    labels = ["positive", "negative", "neutral"] * 5
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LinearSVC())]).fit(texts, labels)
    p = SentimentPredictor(pipe)
    assert not p.supports_proba and p.score_kind == "decision_score"
    result = p.predict_one("good great")
    assert result.score_kind == "decision_score" and result.label == "positive"
    assert len(result.scores) == 3
