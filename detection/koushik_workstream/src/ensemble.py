"""Ensemble voting logic, consensus aggregation, and evaluation metrics computation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


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
    """Compute predictions and confidences for an entire dataset."""
    preds = []
    confs = []
    for sample_votes in all_learner_votes:
        label, conf = compute_majority_vote(sample_votes)
        preds.append(label)
        confs.append(conf)
    return preds, confs


def compute_comprehensive_metrics(
    y_true: List[str] | np.ndarray,
    y_pred: List[str] | np.ndarray,
    class_names: List[str],
    train_duration: float = 0.0,
    infer_duration: float = 0.0,
) -> Dict[str, Any]:
    """Compute accuracy, macro/weighted precision, recall, F1, per-class reports, and confusion matrix."""
    y_t = [str(x) for x in y_true]
    y_p = [str(x) for x in y_pred]

    acc = float(accuracy_score(y_t, y_p))
    macro_prec = float(precision_score(y_t, y_p, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_t, y_p, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))

    weighted_prec = float(precision_score(y_t, y_p, average="weighted", zero_division=0))
    weighted_rec = float(recall_score(y_t, y_p, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_t, y_p, average="weighted", zero_division=0))

    report = classification_report(y_t, y_p, labels=class_names, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_t, y_p, labels=class_names).tolist()

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_prec, 4),
        "weighted_recall": round(weighted_rec, 4),
        "weighted_f1": round(weighted_f1, 4),
        "train_duration_seconds": round(train_duration, 3),
        "infer_duration_seconds": round(infer_duration, 3),
        "classification_report": report,
        "confusion_matrix": cm,
        "class_names": class_names,
    }
