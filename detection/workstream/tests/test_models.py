"""Tests for ensemble voting logic, evaluation metrics, and detector model training.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ensemble import (
    LearnerVote,
    compute_majority_vote,
    compute_comprehensive_metrics,
)
from src.preprocessing import prepare_data_and_splits
from src.train_traditional import train_traditional_models
from src.train_dl import train_dl_models, TORCH_AVAILABLE


def test_majority_voting_agreement():
    # 2 agree, 1 dissents
    votes = [
        LearnerVote(learner_name="RF", voted_label="Benign", learner_confidence=0.85),
        LearnerVote(learner_name="XGB", voted_label="Benign", learner_confidence=0.95),
        LearnerVote(learner_name="MLP", voted_label="DDoS", learner_confidence=0.60),
    ]
    pred_label, conf = compute_majority_vote(votes)
    assert pred_label == "Benign"
    assert round(conf, 2) == 0.90  # average of 0.85 and 0.95


def test_majority_voting_unanimous():
    # All 3 agree
    votes = [
        LearnerVote(learner_name="CNN", voted_label="PortScan", learner_confidence=0.80),
        LearnerVote(learner_name="LSTM", voted_label="PortScan", learner_confidence=0.85),
        LearnerVote(learner_name="Transformer", voted_label="PortScan", learner_confidence=0.90),
    ]
    pred_label, conf = compute_majority_vote(votes)
    assert pred_label == "PortScan"
    assert round(conf, 2) == 0.85


def test_majority_voting_three_way_tie_broken_by_max_confidence():
    # 3-way tie: each model predicts a different label
    votes = [
        LearnerVote(learner_name="Learner1", voted_label="ClassA", learner_confidence=0.65),
        LearnerVote(learner_name="Learner2", voted_label="ClassB", learner_confidence=0.92),
        LearnerVote(learner_name="Learner3", voted_label="ClassC", learner_confidence=0.70),
    ]
    pred_label, conf = compute_majority_vote(votes)
    assert pred_label == "ClassB"
    assert conf == 0.92


def test_compute_comprehensive_metrics():
    y_true = ["A", "B", "C", "A"]
    y_pred = ["A", "B", "B", "A"]
    classes = ["A", "B", "C"]

    metrics = compute_comprehensive_metrics(y_true, y_pred, classes, train_duration=1.2, infer_duration=0.1)
    assert metrics["accuracy"] == 0.75
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["train_duration_seconds"] == 1.2


def test_train_traditional_models_runs(temp_workstream_env):
    cfg = temp_workstream_env["config"]
    df = temp_workstream_env["df"]

    splits = prepare_data_and_splits(df, cfg)
    res = train_traditional_models(splits, cfg)

    assert "models" in res
    assert "rf" in res["models"]
    assert "xgb" in res["models"]
    assert "mlp" in res["models"]
    assert res["metrics"]["ensemble"]["accuracy"] > 0.0
    assert len(res["votes"]) == len(splits.y_test)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_train_dl_models_runs(temp_workstream_env):
    cfg = temp_workstream_env["config"]
    df = temp_workstream_env["df"]

    splits = prepare_data_and_splits(df, cfg)
    res = train_dl_models(splits, cfg)

    assert "models" in res
    assert "CNN1D" in res["models"]
    assert "LSTM" in res["models"]
    assert "Transformer" in res["models"]
    assert res["metrics"]["ensemble"]["accuracy"] > 0.0
    assert len(res["votes"]) == len(splits.y_test)
