"""Tests unitaires du pipeline ML."""
import os

from model_pipeline import prepare_data, train_model, save_model, load_model

DATA = "Churn_Modelling.csv"


def test_prepare_data():
    x_train, x_test, y_train, y_test = prepare_data(DATA)
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert len(x_train) > len(x_test)


def test_save_and_load_model(tmp_path):
    x_train, _, y_train, _ = prepare_data(DATA)
    model = train_model(x_train, y_train)
    path = str(tmp_path / "model.joblib")
    save_model(model, path)
    assert os.path.exists(path)
    assert load_model(path) is not None
