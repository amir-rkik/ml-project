from model_pipeline import prepare_data


def test_prepare_data():
    x_train, x_test, y_train, y_test = prepare_data(
        "Churn_Modelling (1).csv"
    )

    assert x_train is not None
    assert x_test is not None
    assert y_train is not None
    assert y_test is not None

    assert len(x_train) > 0
    assert len(x_test) > 0
