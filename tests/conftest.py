"""Shared fixtures. Synthetic data here tests SOFTWARE behaviour, never model quality."""
import pytest

from src.demo_data import make_synthetic_reviews
from src.predict import SentimentPredictor
from src.train import train_and_evaluate


@pytest.fixture(scope="session")
def synthetic_df():
    return make_synthetic_reviews(n_per_class=60, seed=1)


@pytest.fixture(scope="session")
def training_result(synthetic_df):
    return train_and_evaluate(synthetic_df, dataset_name="demo_test", seed=7)


@pytest.fixture(scope="session")
def predictor(training_result):
    return SentimentPredictor(training_result.pipeline)


@pytest.fixture()
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTIMENTAI_ARTIFACT_ROOT", str(tmp_path))
    return tmp_path
