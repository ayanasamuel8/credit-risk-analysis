import pytest
import pandas as pd
import numpy as np
import os
import joblib
import tempfile
import shutil
from unittest import mock

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.train import train_model, evaluate_model, save_model

# --------------------------
# Fixtures
# --------------------------

@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def tmp_model_dir():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)

# --------------------------
# Test: train_model
# --------------------------

@mock.patch("src.train.mlflow")  # Replace with your module name
def test_train_model(mock_mlflow, sample_data):
    X_train, X_test, y_train, y_test = sample_data

    mock_mlflow.active_run.return_value = None
    mock_mlflow.start_run.return_value.__enter__.return_value.info.run_id = "fake_run_id"

    model, run_id = train_model(X_train, y_train, model_name="logistic")

    assert hasattr(model, "predict")
    assert run_id == "fake_run_id"
    mock_mlflow.log_params.assert_called()  # ensure params were logged

# --------------------------
# Test: evaluate_model
# --------------------------

@mock.patch("src.train.mlflow")  # Replace with your module name
def test_evaluate_model(mock_mlflow, sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model, _ = train_model(X_train, y_train, model_name="random_forest")

    metrics = evaluate_model(model, X_test, y_test)

    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert all(0 <= v <= 1 for v in metrics.values())
    mock_mlflow.log_metric.assert_called()

# --------------------------
# Test: save_model
# --------------------------

def test_save_model(tmp_model_dir, sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model, _ = train_model(X_train, y_train)

    save_path = os.path.join(tmp_model_dir, "test_model.pkl")
    save_model(model, path=save_path)

    assert os.path.exists(save_path)
    loaded_model = joblib.load(save_path)
    assert hasattr(loaded_model, "predict")
