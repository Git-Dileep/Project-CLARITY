"""Deep Learning Ensemble (1D-CNN, LSTM, and Transformer Encoder) using PyTorch.

Trains three separate deep learning architectures on the exact same train split
and evaluates on the exact same test split with majority-voting ensemble consensus.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

from .config import WorkstreamConfig
from .data_loader import load_dataset
from .preprocessing import prepare_data_and_splits, SplitData
from .ensemble import LearnerVote, compute_ensemble_predictions, compute_comprehensive_metrics

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TabularFlowDataset(Dataset):
    """Dataset wrapper for flow feature vectors."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class CNN1DModel(nn.Module):
    """One-dimensional Convolutional Neural Network for flow classification."""
    def __init__(self, n_features: int, n_classes: int, channels: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(channels * 2, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, n_features) -> (batch_size, 1, n_features)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feat = self.conv(x)
        return self.classifier(feat)


class LSTMModel(nn.Module):
    """Bidirectional LSTM network for flow feature sequences."""
    def __init__(self, n_features: int, n_classes: int, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, n_features) -> (batch_size, 1, n_features)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.classifier(last_step)


class TransformerModel(nn.Module):
    """Transformer Encoder architecture for tabular flow representations."""
    def __init__(self, n_features: int, n_classes: int, d_model: int = 32, nhead: int = 4):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, n_features) -> (batch_size, 1, n_features)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.proj(x)
        encoded = self.transformer(h)
        return self.classifier(encoded[:, 0, :])


def get_device(pref: str) -> torch.device:
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_single_dl_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int = 2,
) -> Tuple[nn.Module, float, List[Dict[str, float]]]:
    """Train single PyTorch model with early stopping and loss tracking."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = []

    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
            train_count += len(yb)
        avg_train_loss = train_loss / max(1, train_count)

        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * len(yb)
                val_count += len(yb)
        avg_val_loss = val_loss / max(1, val_count)

        history.append({"epoch": ep + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
        print(f"  [{name}] Epoch {ep+1:02d}/{epochs:02d} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience and ep >= 2:
                print(f"  [{name}] Early stopping triggered at epoch {ep+1}.")
                break

    train_duration = time.perf_counter() - t0
    model.load_state_dict(best_weights)
    return model, train_duration, history


def evaluate_dl_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Evaluate PyTorch model, returning predicted probabilities, labels, and latency."""
    model.eval()
    all_probs = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    infer_duration = time.perf_counter() - t0
    probs_arr = np.concatenate(all_probs, axis=0)
    preds_arr = np.argmax(probs_arr, axis=1)
    return probs_arr, preds_arr, infer_duration


def train_dl_models(
    split_data: SplitData,
    config: WorkstreamConfig,
    base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train 1D-CNN, LSTM, and Transformer models, and combine into a majority-voting ensemble."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for deep learning ensemble. Install via: pip install torch")

    work_dir = base_dir or Path.cwd()
    reports_dl_dir = config.resolve_path(work_dir, config.reports_dir) / "dl"
    models_dir = config.resolve_path(work_dir, config.models_dir)
    reports_dl_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    device = get_device(config.dl.device)
    print(f"\n[DEEP LEARNING] PyTorch Version: {torch.__version__} | Executing on: {device}")

    X_train = split_data.X_train
    y_train = split_data.y_train
    X_test = split_data.X_test
    y_test = split_data.y_test
    classes = split_data.class_names
    n_features = X_train.shape[1]
    n_classes = len(classes)

    # Sub-split train into train/validation for early stopping
    val_size = int(len(X_train) * 0.15)
    train_ds = TabularFlowDataset(X_train[:-val_size], y_train[:-val_size])
    val_ds = TabularFlowDataset(X_train[-val_size:], y_train[-val_size:])
    test_ds = TabularFlowDataset(X_test, y_test)

    batch_size = config.dl.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Instantiate the 3 required models
    models = {
        "CNN1D": CNN1DModel(n_features=n_features, n_classes=n_classes, channels=config.dl.cnn_channels),
        "LSTM": LSTMModel(n_features=n_features, n_classes=n_classes, hidden_size=config.dl.lstm_hidden),
        "Transformer": TransformerModel(n_features=n_features, n_classes=n_classes, d_model=config.dl.transformer_dim, nhead=config.dl.transformer_heads),
    }

    trained_models: Dict[str, nn.Module] = {}
    train_durations: Dict[str, float] = {}
    infer_durations: Dict[str, float] = {}
    model_probs: Dict[str, np.ndarray] = {}
    model_preds: Dict[str, List[str]] = {}
    model_confs: Dict[str, np.ndarray] = {}
    training_histories: Dict[str, List[Dict[str, float]]] = {}

    for name, model in models.items():
        print(f"\n[DEEP LEARNING] Training {name}...")
        fitted_model, tr_time, history = train_single_dl_model(
            name=name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=config.dl.epochs,
            lr=config.dl.learning_rate,
            weight_decay=config.dl.weight_decay,
            patience=config.dl.early_stopping_patience,
        )
        trained_models[name] = fitted_model
        train_durations[name] = tr_time
        training_histories[name] = history

        probs, preds_idx, inf_time = evaluate_dl_model(fitted_model, test_loader, device)
        model_probs[name] = probs
        infer_durations[name] = inf_time
        model_preds[name] = [classes[i] for i in preds_idx]
        model_confs[name] = np.max(probs, axis=1)

    # Build 3 learner votes per test sample
    all_votes: List[List[LearnerVote]] = []
    for i in range(len(X_test)):
        sample_votes = [
            LearnerVote(learner_name="CNN1D", voted_label=model_preds["CNN1D"][i], learner_confidence=float(model_confs["CNN1D"][i])),
            LearnerVote(learner_name="LSTM", voted_label=model_preds["LSTM"][i], learner_confidence=float(model_confs["LSTM"][i])),
            LearnerVote(learner_name="Transformer", voted_label=model_preds["Transformer"][i], learner_confidence=float(model_confs["Transformer"][i])),
        ]
        all_votes.append(sample_votes)

    ensemble_preds, ensemble_confs = compute_ensemble_predictions(all_votes)
    true_labels = [classes[i] for i in y_test]

    # Metrics
    total_tr_time = sum(train_durations.values())
    total_inf_time = sum(infer_durations.values())
    ensemble_metrics = compute_comprehensive_metrics(true_labels, ensemble_preds, classes, total_tr_time, total_inf_time)
    cnn_metrics = compute_comprehensive_metrics(true_labels, model_preds["CNN1D"], classes, train_durations["CNN1D"], infer_durations["CNN1D"])
    lstm_metrics = compute_comprehensive_metrics(true_labels, model_preds["LSTM"], classes, train_durations["LSTM"], infer_durations["LSTM"])
    trans_metrics = compute_comprehensive_metrics(true_labels, model_preds["Transformer"], classes, train_durations["Transformer"], infer_durations["Transformer"])

    # Export test predictions
    test_df = split_data.test_df.copy()
    test_df["true_label"] = true_labels
    test_df["ensemble_prediction"] = ensemble_preds
    test_df["ensemble_confidence"] = ensemble_confs
    test_df["cnn_prediction"] = model_preds["CNN1D"]
    test_df["cnn_confidence"] = model_confs["CNN1D"]
    test_df["lstm_prediction"] = model_preds["LSTM"]
    test_df["lstm_confidence"] = model_confs["LSTM"]
    test_df["transformer_prediction"] = model_preds["Transformer"]
    test_df["transformer_confidence"] = model_confs["Transformer"]
    test_df["ensemble_correct"] = test_df["true_label"] == test_df["ensemble_prediction"]

    pred_export_cols = [
        c for c in [
            "sample_id", "flow_id", "timestamp", "device_id", "true_label",
            "ensemble_prediction", "ensemble_confidence",
            "cnn_prediction", "cnn_confidence",
            "lstm_prediction", "lstm_confidence",
            "transformer_prediction", "transformer_confidence",
            "ensemble_correct"
        ] if c in test_df.columns
    ]
    test_predictions_path = reports_dl_dir / "dl_test_predictions.csv"
    test_df[pred_export_cols].to_csv(test_predictions_path, index=False)

    # Save reports
    pd.DataFrame(ensemble_metrics["classification_report"]).T.to_csv(
        reports_dl_dir / "dl_classification_report.csv", index=True, index_label="class"
    )
    pd.DataFrame(
        ensemble_metrics["confusion_matrix"], index=classes, columns=classes
    ).to_csv(reports_dl_dir / "dl_confusion_matrix.csv")

    summary_metrics = {
        "ensemble": ensemble_metrics,
        "learners": {
            "CNN1D": cnn_metrics,
            "LSTM": lstm_metrics,
            "Transformer": trans_metrics,
        },
        "training_history": training_histories,
        "config": {
            "dataset_name": config.dataset_name,
            "is_synthetic": config.is_synthetic,
            "device": str(device),
            "features_used": split_data.feature_names,
            "classes": classes,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }
    }

    with (reports_dl_dir / "dl_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    metric_rows = [
        {"model": "DL Ensemble (Hard Voting)", **{k: ensemble_metrics[k] for k in ["accuracy", "macro_f1", "weighted_f1", "train_duration_seconds", "infer_duration_seconds"]}},
        {"model": "CNN1D", **{k: cnn_metrics[k] for k in ["accuracy", "macro_f1", "weighted_f1", "train_duration_seconds", "infer_duration_seconds"]}},
        {"model": "LSTM", **{k: lstm_metrics[k] for k in ["accuracy", "macro_f1", "weighted_f1", "train_duration_seconds", "infer_duration_seconds"]}},
        {"model": "Transformer", **{k: trans_metrics[k] for k in ["accuracy", "macro_f1", "weighted_f1", "train_duration_seconds", "infer_duration_seconds"]}},
    ]
    pd.DataFrame(metric_rows).to_csv(reports_dl_dir / "dl_metrics.csv", index=False)

    # Save checkpoints locally
    checkpoint = {
        "cnn_state": trained_models["CNN1D"].state_dict(),
        "lstm_state": trained_models["LSTM"].state_dict(),
        "transformer_state": trained_models["Transformer"].state_dict(),
        "classes": classes,
        "n_features": n_features,
        "feature_names": split_data.feature_names,
    }
    torch.save(checkpoint, models_dir / "dl_ensemble.pt")

    print("\n[DEEP LEARNING] Ensemble Training Completed Successfully!")
    print(f"  DL Ensemble Accuracy: {ensemble_metrics['accuracy']:.4f}")
    print(f"  DL Ensemble Macro F1: {ensemble_metrics['macro_f1']:.4f}")
    print(f"  DL Ensemble Weighted F1: {ensemble_metrics['weighted_f1']:.4f}")
    print(f"  Reports saved to {reports_dl_dir}")

    return {
        "models": trained_models,
        "test_df": test_df,
        "metrics": summary_metrics,
        "votes": all_votes,
    }


def main():
    parser = argparse.ArgumentParser(description="Train deep learning 3-model ensemble.")
    parser.add_argument("--config", default="config/reference.json", help="Path to config file")
    args = parser.parse_args()

    cfg = WorkstreamConfig.load(args.config)
    df, _ = load_dataset(cfg)
    splits = prepare_data_and_splits(df, cfg)
    train_dl_models(splits, cfg)


if __name__ == "__main__":
    main()
