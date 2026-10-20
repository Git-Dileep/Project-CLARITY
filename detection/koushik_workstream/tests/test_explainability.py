"""Tests for SHAP explainability, top-10 attribution ranking, and contract format.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.explainability import (
    get_top_10_attributions,
    explain_traditional_tree,
    explain_dl_gradient,
)
from src.preprocessing import prepare_data_and_splits
from src.train_traditional import train_traditional_models
from src.train_dl import train_dl_models, TORCH_AVAILABLE


def test_get_top_10_attributions_ranking():
    feat_names = [f"feat_{i}" for i in range(15)]
    shap_vals = np.array([0.1, -0.9, 0.4, 0.05, -0.7, 0.85, 0.01, -0.3, 0.2, 0.15, -0.02, 0.6, -0.5, 0.08, 0.03])
    feat_vals = np.arange(15, dtype=float)

    top_10 = get_top_10_attributions(shap_vals, feat_vals, feat_names)

    assert len(top_10) == 10
    # Check ranks 1..10
    ranks = [item["rank"] for item in top_10]
    assert ranks == list(range(1, 11))

    # Check absolute descending order
    abs_shaps = [abs(item["shap_value"]) for item in top_10]
    assert abs_shaps == sorted(abs_shaps, reverse=True)

    # First should be feat_1 (abs value 0.9)
    assert top_10[0]["feature_name"] == "feat_1"
    assert top_10[0]["shap_value"] == -0.9


def test_get_top_10_attributions_fails_when_under_10_features():
    feat_names = [f"feat_{i}" for i in range(8)]
    shap_vals = np.random.randn(8)
    feat_vals = np.arange(8, dtype=float)

    with pytest.raises(ValueError, match="requires exactly 10 features, but only 8"):
        get_top_10_attributions(shap_vals, feat_vals, feat_names)


def test_traditional_tree_explainer(temp_workstream_env):
    cfg = temp_workstream_env["config"]
    df = temp_workstream_env["df"]

    splits = prepare_data_and_splits(df, cfg)
    res = train_traditional_models(splits, cfg)

    xgb_model = res["models"]["xgb"]
    sample_X = splits.X_test[:2]
    preds = [splits.class_names[0], splits.class_names[1]]

    exps = explain_traditional_tree(
        model=xgb_model,
        X_samples=sample_X,
        feature_names=splits.feature_names,
        class_names=splits.class_names,
        predicted_classes=preds,
    )

    assert len(exps) == 2
    for exp in exps:
        assert exp["explainer_type"] == "TreeExplainer"
        assert len(exp["top_features"]) == 10
        assert "base_value" in exp
        assert "explained_class" in exp


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_dl_gradient_explainer(temp_workstream_env):
    cfg = temp_workstream_env["config"]
    df = temp_workstream_env["df"]

    splits = prepare_data_and_splits(df, cfg)
    res = train_dl_models(splits, cfg)

    transformer_model = res["models"]["Transformer"]
    bg = splits.X_train[:10]
    sample_X = splits.X_test[:2]
    preds = [splits.class_names[0], splits.class_names[1]]

    exps = explain_dl_gradient(
        model=transformer_model,
        background_samples=bg,
        X_samples=sample_X,
        feature_names=splits.feature_names,
        class_names=splits.class_names,
        predicted_classes=preds,
    )

    assert len(exps) == 2
    for exp in exps:
        assert exp["explainer_type"] == "GradientExplainer"
        assert len(exp["top_features"]) == 10
