"""Explainability module for computing exact top-10 SHAP feature attributions for alerts.

Supports:
- TreeExplainer for traditional tree-based models (XGBoost).
- GradientExplainer for PyTorch deep learning models (Transformer / CNN).
- Strict adherence to contracts/alert_schema_v1.json:
  Exactly 10 features, ranked 1..10 by descending abs(shap_value).
"""

from __future__ import annotations

import argparse
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import shap
import torch

from .config import WorkstreamConfig


def get_top_10_attributions(
    shap_vals: np.ndarray,
    feat_vals: np.ndarray,
    feature_names: List[str],
) -> List[Dict[str, Any]]:
    """Sort attributions by descending absolute SHAP magnitude and return top 10 ranked features."""
    if len(feature_names) < 10:
        raise ValueError(
            f"Contract requires exactly 10 features, but only {len(feature_names)} features exist in model input."
        )

    # Flatten if multi-dimensional (e.g. sequence window or channel)
    s_vals = np.asarray(shap_vals, dtype=np.float64).flatten()
    f_vals = np.asarray(feat_vals, dtype=np.float64).flatten()

    if len(s_vals) != len(feature_names):
        s_vals = s_vals[:len(feature_names)]
        f_vals = f_vals[:len(feature_names)]

    # Sort indices by descending absolute attribution with stable index tie-breaking
    sorted_indices = sorted(range(len(s_vals)), key=lambda idx: (-abs(s_vals[idx]), idx))[:10]

    top_features: List[Dict[str, Any]] = []
    for rank, idx in enumerate(sorted_indices, start=1):
        top_features.append({
            "rank": int(rank),
            "feature_name": str(feature_names[idx]),
            "feature_value": float(round(f_vals[idx], 4)),
            "shap_value": float(round(s_vals[idx], 6)),
        })
    return top_features


def explain_traditional_tree(
    model: Any,
    X_samples: np.ndarray,
    feature_names: List[str],
    class_names: List[str],
    predicted_classes: List[str],
) -> List[Dict[str, Any]]:
    """Generate SHAP TreeExplainer explanations for traditional detector."""
    explainer = shap.TreeExplainer(model)
    raw_shap_values = explainer.shap_values(X_samples)

    # Expected value baseline
    exp_val = explainer.expected_value
    if isinstance(exp_val, (list, np.ndarray)):
        base_vals = [float(x) for x in exp_val]
    else:
        base_vals = [float(exp_val)] * len(class_names)

    explanations: List[Dict[str, Any]] = []

    for i in range(len(X_samples)):
        pred_label = predicted_classes[i]
        class_idx = class_names.index(pred_label) if pred_label in class_names else 0
        base_value = base_vals[class_idx] if class_idx < len(base_vals) else 0.0

        if isinstance(raw_shap_values, list):
            # List of classes -> (n_samples, n_features)
            sample_shap = raw_shap_values[class_idx][i]
        elif isinstance(raw_shap_values, np.ndarray):
            if raw_shap_values.ndim == 3:
                # (n_samples, n_features, n_classes) or (n_samples, n_classes, n_features)
                if raw_shap_values.shape[2] == len(class_names):
                    sample_shap = raw_shap_values[i, :, class_idx]
                else:
                    sample_shap = raw_shap_values[i, class_idx, :]
            else:
                sample_shap = raw_shap_values[i]
        else:
            sample_shap = np.asarray(raw_shap_values)[i]

        top_feats = get_top_10_attributions(
            shap_vals=sample_shap,
            feat_vals=X_samples[i],
            feature_names=feature_names,
        )

        explanations.append({
            "explainer_type": "TreeExplainer",
            "explained_class": str(pred_label),
            "base_value": float(round(base_value, 4)),
            "top_features": top_feats,
        })

    return explanations


def explain_dl_gradient(
    model: torch.nn.Module,
    background_samples: np.ndarray,
    X_samples: np.ndarray,
    feature_names: List[str],
    class_names: List[str],
    predicted_classes: List[str],
    device: torch.device = torch.device("cpu"),
) -> List[Dict[str, Any]]:
    """Generate SHAP GradientExplainer explanations for deep learning model."""
    model.eval()
    bg_tensor = torch.tensor(background_samples, dtype=torch.float32).to(device)
    ex_tensor = torch.tensor(X_samples, dtype=torch.float32).to(device)

    explainer = shap.GradientExplainer(model, bg_tensor)
    raw_shap_values = explainer.shap_values(ex_tensor)

    # Compute empirical base values from model predictions on background
    with torch.no_grad():
        bg_out = torch.softmax(model(bg_tensor), dim=1).mean(dim=0).cpu().numpy()

    explanations: List[Dict[str, Any]] = []

    for i in range(len(X_samples)):
        pred_label = predicted_classes[i]
        class_idx = class_names.index(pred_label) if pred_label in class_names else 0
        base_value = float(bg_out[class_idx]) if class_idx < len(bg_out) else 0.0

        if isinstance(raw_shap_values, list):
            sample_shap = np.asarray(raw_shap_values[class_idx])[i]
        elif isinstance(raw_shap_values, np.ndarray):
            if raw_shap_values.ndim == 3:
                sample_shap = raw_shap_values[i, :, class_idx] if raw_shap_values.shape[2] == len(class_names) else raw_shap_values[i, class_idx, :]
            elif raw_shap_values.ndim == 4:
                sample_shap = raw_shap_values[i, :, :, class_idx].mean(axis=0)
            else:
                sample_shap = raw_shap_values[i]
        else:
            sample_shap = np.asarray(raw_shap_values)[i]

        top_feats = get_top_10_attributions(
            shap_vals=sample_shap,
            feat_vals=X_samples[i],
            feature_names=feature_names,
        )

        explanations.append({
            "explainer_type": "GradientExplainer",
            "explained_class": str(pred_label),
            "base_value": float(round(base_value, 4)),
            "top_features": top_feats,
        })

    return explanations
