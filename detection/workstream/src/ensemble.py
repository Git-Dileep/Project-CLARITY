"""Ensemble voting logic, consensus aggregation, and evaluation metrics computation.
Supports both Hard Voting (majority rule with confidence tie-breaking) and
Soft Voting (uniform continuous posterior probability averaging).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


@dataclass
class LearnerVote:
    learner_name: str
    voted_label: str
    learner_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learner_name": self.learner_name,
            "voted_label": str(self.voted_label),
            "learner_confidence": round(float(self.learner_confidence), 4),
        }


def assert_class_alignment(
    learner_classes_dict: Dict[str, List[Any] | np.ndarray],
    expected_classes: List[str],
) -> None:
    """Strictly assert that every base learner shares the exact same class labels in identical order.
    
    Prevents silent column permutation errors during probability averaging.
    """
    exp = list(expected_classes)
    for model_name, cls_list in learner_classes_dict.items():
        actual = [str(c) for c in cls_list]
        if actual != exp:
            raise ValueError(
                f"Class alignment violation in {model_name}!\n"
                f"Expected ({len(exp)} classes): {exp}\n"
                f"Actual   ({len(actual)} classes): {actual}"
            )


def compute_majority_vote(
    sample_votes: List[LearnerVote],
) -> Tuple[str, float]:
    """Compute consensus predicted label and ensemble confidence across 3 base learners.

    Uses majority vote. If all 3 learners vote differently (1-1-1 split),
    the tie is broken by the learner with the highest confidence.
    Ensemble confidence is the average confidence of the agreeing learners.
    """
    counts = Counter(v.voted_label for v in sample_votes)
    most_common = counts.most_common()

    if len(most_common) == 1:
        # All 3 agree
        consensus_label = most_common[0][0]
        agreeing_confs = [v.learner_confidence for v in sample_votes if v.voted_label == consensus_label]
        consensus_conf = float(np.mean(agreeing_confs))
    elif most_common[0][1] >= 2:
        # 2 out of 3 agree
        consensus_label = most_common[0][0]
        agreeing_confs = [v.learner_confidence for v in sample_votes if v.voted_label == consensus_label]
        consensus_conf = float(np.mean(agreeing_confs))
    else:
        # 3-way tie (1-1-1): break tie with highest confidence learner
        best_vote = max(sample_votes, key=lambda v: v.learner_confidence)
        consensus_label = best_vote.voted_label
        consensus_conf = float(best_vote.learner_confidence)

    return consensus_label, float(np.clip(consensus_conf, 0.0, 1.0))


def compute_ensemble_predictions(
    all_learner_votes: List[List[LearnerVote]],
) -> Tuple[List[str], List[float]]:
    """Compute predictions and confidences for an entire dataset using Hard Voting."""
    preds = []
    confs = []
    for sample_votes in all_learner_votes:
        label, conf = compute_majority_vote(sample_votes)
        preds.append(label)
        confs.append(conf)
    return preds, confs


def compute_soft_voting_predictions(
    learner_probs: Dict[str, np.ndarray],
    class_names: List[str],
) -> Tuple[List[str], List[float], np.ndarray]:
    """Compute predictions and confidences using uniform posterior probability averaging (Soft Voting).
    
    Averages posterior probability distributions across learners without weights or calibration:
    P_soft = (1/M) * sum(P_m)
    """
    prob_list = list(learner_probs.values())
    if not prob_list:
        raise ValueError("No learner probabilities provided for soft voting.")

    # Validate shape and columns
    n_samples, n_classes = prob_list[0].shape
    if n_classes != len(class_names):
        raise ValueError(
            f"Probability column count ({n_classes}) does not match class_names length ({len(class_names)})."
        )

    for name, p in learner_probs.items():
        if p.shape != (n_samples, n_classes):
            raise ValueError(
                f"Shape mismatch for {name}: expected {(n_samples, n_classes)}, got {p.shape}"
            )

    # Uniform mean probability across learners
    stacked = np.stack(prob_list, axis=0)  # (M, N, C)
    mean_probs = np.mean(stacked, axis=0)   # (N, C)

    soft_preds_idx = np.argmax(mean_probs, axis=1)
    soft_confs = np.max(mean_probs, axis=1)
    soft_preds = [class_names[idx] for idx in soft_preds_idx]

    return soft_preds, [float(c) for c in soft_confs], mean_probs


def compute_voting_discordance(
    hard_preds: List[str],
    soft_preds: List[str],
) -> Dict[str, Any]:
    """Calculate sample-level agreement and disagreement metrics between Hard and Soft Voting."""
    if len(hard_preds) != len(soft_preds):
        raise ValueError("hard_preds and soft_preds must have identical length.")

    n_total = len(hard_preds)
    discordant_mask = [h != s for h, s in zip(hard_preds, soft_preds)]
    discordant_count = int(sum(discordant_mask))
    discordance_rate = float(discordant_count / max(1, n_total))

    return {
        "total_samples": n_total,
        "agree_count": n_total - discordant_count,
        "disagree_count": discordant_count,
        "disagreement_rate": round(discordance_rate, 4),
        "disagreement_percentage": round(discordance_rate * 100.0, 2),
    }


def compute_mcnemar_test(
    y_true: List[str],
    y_pred_hard: List[str],
    y_pred_soft: List[str],
) -> Dict[str, Any]:
    """Calculate McNemar paired contingency test on classification correctness."""
    hard_correct = np.array([yt == yp for yt, yp in zip(y_true, y_pred_hard)])
    soft_correct = np.array([yt == yp for yt, yp in zip(y_true, y_pred_soft)])

    n00 = int(np.sum(hard_correct & soft_correct))          # Both correct
    n01 = int(np.sum(hard_correct & (~soft_correct)))       # Hard correct, Soft wrong (n_HardOnly)
    n10 = int(np.sum((~hard_correct) & soft_correct))       # Soft correct, Hard wrong (n_SoftOnly)
    n11 = int(np.sum((~hard_correct) & (~soft_correct)))    # Both wrong

    discordant_total = n01 + n10
    if discordant_total == 0:
        statistic = 0.0
        p_value = 1.0
        test_type = "exact_binomial (zero discordant)"
    elif discordant_total < 25:
        # Exact two-sided binomial test
        p_value = float(stats.binomtest(min(n01, n10), discordant_total, 0.5).pvalue)
        statistic = float(min(n01, n10))
        test_type = "exact_binomial"
    else:
        # Edwards continuity-corrected chi-square
        statistic = float((abs(n01 - n10) - 1.0) ** 2 / discordant_total)
        p_value = float(stats.chi2.sf(statistic, df=1))
        test_type = "mcnemar_chi2_continuity_corrected"

    return {
        "test_type": test_type,
        "contingency_table": {
            "n_both_correct": n00,
            "n_HardOnly_correct": n01,
            "n_SoftOnly_correct": n10,
            "n_both_incorrect": n11,
        },
        "statistic": round(statistic, 4),
        "p_value": round(p_value, 6),
        "significant_alpha_0_025": bool(p_value < 0.025),
        "significant_alpha_0_05": bool(p_value < 0.05),
    }


def compute_paired_bootstrap_macro_f1(
    y_true: List[str],
    y_pred_hard: List[str],
    y_pred_soft: List[str],
    eval_classes: List[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Perform pre-specified paired bootstrap test of the macro-F1 difference (Soft minus Hard).
    
    Evaluates over the fixed 11-class set (excluding ARP Spoofing).
    Resamples test flows in pairs with fixed seed.
    """
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)

    yt_arr = np.array(y_true)
    yh_arr = np.array(y_pred_hard)
    ys_arr = np.array(y_pred_soft)

    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        f1_h = f1_score(yt_arr[idx], yh_arr[idx], labels=eval_classes, average="macro", zero_division=0)
        f1_s = f1_score(yt_arr[idx], ys_arr[idx], labels=eval_classes, average="macro", zero_division=0)
        diffs.append(f1_s - f1_h)

    diffs = np.array(diffs)
    mean_diff = float(np.mean(diffs))
    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))

    # Two-sided empirical p-value for difference != 0
    p_val = float(2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
    p_val = min(1.0, max(0.0, p_val))

    return {
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "eval_classes_count": len(eval_classes),
        "mean_difference_soft_minus_hard": round(mean_diff, 6),
        "ci_95_lower": round(ci_lower, 6),
        "ci_95_upper": round(ci_upper, 6),
        "p_value_empirical": round(p_val, 6),
        "significant_bonferroni_0_025": bool(p_val < 0.025),
    }


def compute_comprehensive_metrics(
    y_true: List[str] | np.ndarray,
    y_pred: List[str] | np.ndarray,
    class_names: List[str],
    train_duration: float = 0.0,
    infer_duration: float = 0.0,
    excluded_from_primary: str = "ARP Spoofing",
) -> Dict[str, Any]:
    """Compute pre-specified metrics:
    - Primary: 11-class macro-F1 (excluding ARP Spoofing).
    - Secondary: 12-class macro-F1 (flagged high-variance), accuracy, weighted F1, per-class report.
    - Descriptive ARP Spoofing row.
    """
    y_t = [str(x) for x in y_true]
    y_p = [str(x) for x in y_pred]

    acc = float(accuracy_score(y_t, y_p))

    # 12-class secondary metrics
    macro_prec_12 = float(precision_score(y_t, y_p, labels=class_names, average="macro", zero_division=0))
    macro_rec_12 = float(recall_score(y_t, y_p, labels=class_names, average="macro", zero_division=0))
    macro_f1_12 = float(f1_score(y_t, y_p, labels=class_names, average="macro", zero_division=0))

    weighted_prec = float(precision_score(y_t, y_p, labels=class_names, average="weighted", zero_division=0))
    weighted_rec = float(recall_score(y_t, y_p, labels=class_names, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_t, y_p, labels=class_names, average="weighted", zero_division=0))

    # Pre-specified primary 11-class metrics (excluding ARP Spoofing)
    primary_classes = [c for c in class_names if c != excluded_from_primary]
    macro_prec_11 = float(precision_score(y_t, y_p, labels=primary_classes, average="macro", zero_division=0))
    macro_rec_11 = float(recall_score(y_t, y_p, labels=primary_classes, average="macro", zero_division=0))
    macro_f1_11 = float(f1_score(y_t, y_p, labels=primary_classes, average="macro", zero_division=0))

    report = classification_report(y_t, y_p, labels=class_names, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_t, y_p, labels=class_names).tolist()

    # Descriptive ARP Spoofing metrics
    arp_desc = report.get(excluded_from_primary, {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})

    return {
        # Primary pre-specified headline metric (11 classes)
        "macro_f1_11class_primary": round(macro_f1_11, 4),
        "macro_precision_11class": round(macro_prec_11, 4),
        "macro_recall_11class": round(macro_rec_11, 4),

        # Secondary metrics
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1_12, 4),  # Standard 12-class for backward compatibility
        "macro_f1_12class_secondary": round(macro_f1_12, 4),
        "macro_precision": round(macro_prec_12, 4),
        "macro_recall": round(macro_rec_12, 4),
        "weighted_precision": round(weighted_prec, 4),
        "weighted_recall": round(weighted_rec, 4),
        "weighted_f1": round(weighted_f1, 4),

        # Descriptive metrics for excluded class
        "excluded_class_descriptive": {
            "class_name": excluded_from_primary,
            "precision": round(float(arp_desc.get("precision", 0.0)), 4),
            "recall": round(float(arp_desc.get("recall", 0.0)), 4),
            "f1": round(float(arp_desc.get("f1-score", 0.0)), 4),
            "support": int(arp_desc.get("support", 0)),
            "note": "Reported descriptively only due to single test flow support.",
        },

        "train_duration_seconds": round(train_duration, 3),
        "infer_duration_seconds": round(infer_duration, 3),
        "classification_report": report,
        "confusion_matrix": cm,
        "class_names": class_names,
        "primary_class_names": primary_classes,
    }
