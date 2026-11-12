"""Tests for ensemble voting logic, evaluation metrics, and detector model training.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ensemble import (
    LearnerVote,
    compute_majority_vote,
    compute_comprehensive_metrics,
    assert_class_alignment,
    compute_soft_voting_predictions,
    compute_voting_discordance,
    compute_mcnemar_test,
    compute_paired_bootstrap_macro_f1,
)
from src.preprocessing import prepare_data_and_splits
from src.train_traditional import train_traditional_models
from src.train_dl import train_dl_models, TORCH_AVAILABLE


def test_soft_voting_probability_averaging():
    """Verify uniform averaging of probability distributions across 3 base learners."""
    class_names = ["ClassA", "ClassB", "ClassC"]
    probs = {
        "model1": np.array([[0.70, 0.20, 0.10], [0.10, 0.80, 0.10]]),
        "model2": np.array([[0.60, 0.30, 0.10], [0.20, 0.70, 0.10]]),
        "model3": np.array([[0.80, 0.10, 0.10], [0.30, 0.60, 0.10]]),
    }
    preds, confs, mean_p = compute_soft_voting_predictions(probs, class_names)
    assert preds == ["ClassA", "ClassB"]
    assert pytest.approx(confs[0], 1e-4) == 0.70
    assert pytest.approx(confs[1], 1e-4) == 0.70
    assert mean_p.shape == (2, 3)


def test_soft_voting_high_confidence_override():
    """Verify that a highly confident single learner overrides marginal 2-model majority."""
    class_names = ["ClassA", "ClassB"]
    probs = {
        "model1": np.array([[0.51, 0.49]]),  # votes ClassA marginally
        "model2": np.array([[0.51, 0.49]]),  # votes ClassA marginally
        "model3": np.array([[0.01, 0.99]]),  # votes ClassB with high confidence
    }
    # Hard voting gives ClassA (2 vs 1)
    hard_votes = [
        LearnerVote("model1", "ClassA", 0.51),
        LearnerVote("model2", "ClassA", 0.51),
        LearnerVote("model3", "ClassB", 0.99),
    ]
    hard_pred, _ = compute_majority_vote(hard_votes)
    assert hard_pred == "ClassA"

    # Soft voting gives ClassB (mean P(B) = (0.49+0.49+0.99)/3 = 0.6567 > P(A) = 0.3433)
    soft_preds, soft_confs, _ = compute_soft_voting_predictions(probs, class_names)
    assert soft_preds[0] == "ClassB"
    assert pytest.approx(soft_confs[0], 1e-3) == 0.657


def test_soft_voting_class_alignment_assertion():
    """Verify that class order mismatch triggers strict assertion error."""
    expected = ["Benign", "DDoS", "PortScan"]
    mismatched = {
        "model1": ["Benign", "DDoS", "PortScan"],
        "model2": ["DDoS", "Benign", "PortScan"],  # out of order!
    }
    with pytest.raises(ValueError, match="Class alignment violation"):
        assert_class_alignment(mismatched, expected)


def test_voting_discordance_calculation():
    """Verify discordance rate calculation."""
    hard_p = ["A", "B", "C", "D"]
    soft_p = ["A", "B", "A", "D"]
    stats = compute_voting_discordance(hard_p, soft_p)
    assert stats["total_samples"] == 4
    assert stats["agree_count"] == 3
    assert stats["disagree_count"] == 1
    assert stats["disagreement_rate"] == 0.25
    assert stats["disagreement_percentage"] == 25.0


def test_mcnemar_paired_test():
    """Verify McNemar contingency table and exact binomial / chi2 test."""
    y_true = ["A", "B", "A", "B", "A"]
    y_hard = ["A", "B", "B", "B", "A"]  # 4 correct, sample 2 wrong
    y_soft = ["A", "B", "A", "A", "A"]  # 4 correct, sample 3 wrong
    res = compute_mcnemar_test(y_true, y_hard, y_soft)
    assert res["contingency_table"]["n_both_correct"] == 3
    assert res["contingency_table"]["n_HardOnly_correct"] == 1
    assert res["contingency_table"]["n_SoftOnly_correct"] == 1
    assert res["p_value"] == 1.0


def test_paired_bootstrap_macro_f1():
    """Verify paired bootstrap calculation on macro-F1 difference."""
    y_true = ["A", "B", "C", "A", "B"] * 20
    y_hard = ["A", "B", "B", "A", "B"] * 20
    y_soft = ["A", "B", "C", "A", "B"] * 20
    eval_classes = ["A", "B", "C"]
    boot_res = compute_paired_bootstrap_macro_f1(y_true, y_hard, y_soft, eval_classes, n_bootstrap=100, seed=42)
    assert "mean_difference_soft_minus_hard" in boot_res
    assert "ci_95_lower" in boot_res
    assert "ci_95_upper" in boot_res
    assert boot_res["mean_difference_soft_minus_hard"] > 0


def test_primary_11class_macro_f1_metric():
    """Verify that primary metric excludes ARP Spoofing while secondary metric includes it."""
    classes = ["ARP Spoofing", "Benign", "DDoS"]
    y_true = ["ARP Spoofing", "Benign", "DDoS", "Benign"]
    y_pred = ["Benign", "Benign", "DDoS", "Benign"]  # missed ARP Spoofing
    metrics = compute_comprehensive_metrics(y_true, y_pred, classes, excluded_from_primary="ARP Spoofing")
    assert "macro_f1_11class_primary" in metrics
    assert "macro_f1_12class_secondary" in metrics
    assert metrics["macro_f1_11class_primary"] == 0.9  # Benign F1=0.8, DDoS F1=1.0 -> macro=0.9
    assert metrics["macro_f1_12class_secondary"] == 0.6  # Penalized by ARP Spoofing F1=0.0 -> (0.8+1.0+0.0)/3 = 0.6
    assert metrics["excluded_class_descriptive"]["class_name"] == "ARP Spoofing"



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
