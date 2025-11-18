import pandas as pd
from predicting_outcomes_in_heart_failure.modeling.explainability import (
    explain_prediction,
    save_shap_waterfall_plot,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def test_explain_prediction_tree_model_returns_explanations(tmp_path):
    """
    Given a simple trained tree-based model and a single input row,
    explain_prediction should:
        - run without errors
        - return a non-empty list of explanations
        - each explanation should contain 'feature', 'value', 'abs_value'
        - all feature names must belong to the input columns
        - respect the top_k limit
    """
    X_train = pd.DataFrame(
        {
            "Age": [50, 60, 55, 45],
            "RestingBP": [120, 140, 130, 110],
            "Cholesterol": [200, 250, 230, 190],
        }
    )
    y_train = [0, 1, 1, 0]

    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )
    model.fit(X_train, y_train)

    X_single = X_train.iloc[[0]]

    top_k = 2
    explanations = explain_prediction(
        model=model,
        X=X_single,
        model_type="random_forest",
        top_k=top_k,
    )

    assert isinstance(explanations, list)
    assert len(explanations) > 0
    assert len(explanations) <= top_k

    input_features = set(X_single.columns.tolist())
    for exp in explanations:
        assert "feature" in exp
        assert "value" in exp
        assert "abs_value" in exp
        assert exp["feature"] in input_features


def test_explain_prediction_logreg_uses_coefficients():
    """
    For a logistic regression model, explain_prediction should use coef_
    and return explanations for the most important features (by |coef|).
    """
    X_train = pd.DataFrame(
        {
            "Age": [50, 60, 55, 45, 52, 58],
            "RestingBP": [120, 140, 130, 110, 125, 135],
            "Cholesterol": [200, 250, 230, 190, 210, 240],
        }
    )
    y_train = [0, 1, 1, 0, 0, 1]

    model = LogisticRegression(
        solver="liblinear",
        random_state=42,
    )
    model.fit(X_train, y_train)

    X_single = X_train.iloc[[0]]

    top_k = 2
    explanations = explain_prediction(
        model=model,
        X=X_single,
        model_type="logreg",
        top_k=top_k,
    )

    assert isinstance(explanations, list)
    assert len(explanations) > 0
    assert len(explanations) <= top_k

    input_features = set(X_single.columns.tolist())
    for exp in explanations:
        assert "feature" in exp
        assert "value" in exp
        assert "abs_value" in exp
        assert exp["feature"] in input_features


def test_save_shap_waterfall_plot_creates_file(tmp_path):
    """
    save_shap_waterfall_plot should create a PNG file on disk
    for a tree-based model when given a valid single row.
    """
    X_train = pd.DataFrame(
        {
            "Age": [50, 60, 55, 45],
            "RestingBP": [120, 140, 130, 110],
            "Cholesterol": [200, 250, 230, 190],
        }
    )
    y_train = [0, 1, 1, 0]

    model = RandomForestClassifier(
        n_estimators=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    X_single = X_train.iloc[[0]]

    output_path = tmp_path / "waterfall_rf.png"

    result_path = save_shap_waterfall_plot(
        model=model,
        X=X_single,
        model_type="random_forest",
        output_path=output_path,
    )

    assert result_path is not None
    assert result_path.exists()
    assert result_path.suffix == ".png"


def test_save_shap_waterfall_plot_non_tree_returns_none(tmp_path):
    """
    For non-tree models (e.g. logistic regression), the helper should
    not try to build a plot and must return None.
    """
    X_train = pd.DataFrame(
        {
            "Age": [50, 60, 55, 45],
            "RestingBP": [120, 140, 130, 110],
            "Cholesterol": [200, 250, 230, 190],
        }
    )
    y_train = [0, 1, 1, 0]

    model = LogisticRegression(
        solver="liblinear",
        random_state=42,
    )
    model.fit(X_train, y_train)

    X_single = X_train.iloc[[0]]

    output_path = tmp_path / "waterfall_logreg.png"

    result_path = save_shap_waterfall_plot(
        model=model,
        X=X_single,
        model_type="logreg",
        output_path=output_path,
    )

    assert result_path is None
    assert not output_path.exists()
