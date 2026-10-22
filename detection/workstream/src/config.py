"""Configuration management for the Detection Workstream.

Supports loading, validating, and saving configuration files in JSON format.
Ensures independence from specific datasets via dynamic schema definitions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, List, Optional, Dict


@dataclass
class TraditionalConfig:
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = 15
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    mlp_hidden_layer_sizes: List[int] = field(default_factory=lambda: [64, 32])
    mlp_max_iter: int = 60
    mlp_early_stopping: bool = True


@dataclass
class DeepLearningConfig:
    enabled: bool = True
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    cnn_channels: int = 32
    lstm_hidden: int = 32
    transformer_dim: int = 32
    transformer_heads: int = 4
    early_stopping_patience: int = 2
    device: str = "auto"  # 'auto', 'cpu', 'cuda'


@dataclass
class ExplainabilityConfig:
    top_n_features: int = 10
    background_samples: int = 50
    traditional_explainer: str = "TreeExplainer"
    dl_explainer: str = "GradientExplainer"


@dataclass
class PilotConfig:
    pilot_size: int = 20
    min_classes: int = 5
    stratified: bool = True


@dataclass
class WorkstreamConfig:
    dataset_name: str = "Reference-Synthetic-Network"
    is_synthetic: bool = True
    input_path: str = "data/reference_synthetic.csv"
    target_column: str = "label"
    flow_id_column: Optional[str] = "flow_id"
    timestamp_column: Optional[str] = "timestamp"
    device_id_column: Optional[str] = "device_id"
    metadata_columns: List[str] = field(default_factory=lambda: ["device_id", "source_file", "sample_id"])
    drop_columns: List[str] = field(default_factory=lambda: ["src_ip", "dst_ip", "flow_id", "timestamp"])
    numeric_columns: Optional[List[str]] = None
    categorical_columns: Optional[List[str]] = None
    max_rows_per_class: Optional[int] = None
    test_size: float = 0.2
    random_seed: int = 42
    chunksize: int = 100000

    traditional: TraditionalConfig = field(default_factory=TraditionalConfig)
    dl: DeepLearningConfig = field(default_factory=DeepLearningConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    pilot: PilotConfig = field(default_factory=PilotConfig)

    run_id: Optional[str] = None
    split_id: Optional[str] = None
    source_sha256: Optional[str] = None

    reports_dir: str = "reports"
    models_dir: str = "models"
    data_dir: str = "data"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkstreamConfig:
        data = data.copy()
        if "traditional" in data and isinstance(data["traditional"], dict):
            data["traditional"] = TraditionalConfig(**data["traditional"])
        if "dl" in data and isinstance(data["dl"], dict):
            data["dl"] = DeepLearningConfig(**data["dl"])
        if "explainability" in data and isinstance(data["explainability"], dict):
            data["explainability"] = ExplainabilityConfig(**data["explainability"])
        if "pilot" in data and isinstance(data["pilot"], dict):
            data["pilot"] = PilotConfig(**data["pilot"])
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> WorkstreamConfig:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    def resolve_path(self, base_dir: Path, rel_path: str) -> Path:
        p = Path(rel_path)
        if p.is_absolute():
            return p
        return base_dir / p
