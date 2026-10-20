"""Leakage-free preprocessing, feature transformation, and deterministic train/test splitting.

Guarantees:
- Preprocessors are fitted ONLY on training split.
- Reusable scikit-learn transformers.
- Categorical one-hot encoding with unknown category handling.
- Deterministic stratified splitting.
- Saves split manifest and prevents metadata/target leakage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from .config import WorkstreamConfig


@dataclass
class SplitData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_names: List[str]
    class_names: List[str]
    preprocessor: FeaturePreprocessor
    label_encoder: LabelEncoder
    split_manifest: Dict[str, Any]


class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Modular feature preprocessor fitting strictly on training data."""

    def __init__(
        self,
        numeric_cols: List[str],
        categorical_cols: List[str],
    ):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.pipeline: Optional[ColumnTransformer] = None
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> FeaturePreprocessor:
        transformers = []

        if self.numeric_cols:
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("numeric", num_pipe, self.numeric_cols))

        if self.categorical_cols:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("categorical", cat_pipe, self.categorical_cols))

        if not transformers:
            raise ValueError("No numeric or categorical features available to fit.")

        self.pipeline = ColumnTransformer(transformers=transformers, remainder="drop")
        self.pipeline.fit(X)

        # Extract output feature names
        names: List[str] = []
        if self.numeric_cols:
            names.extend(self.numeric_cols)
        if self.categorical_cols:
            cat_encoder = self.pipeline.named_transformers_["categorical"].named_steps["encoder"]
            cat_names = cat_encoder.get_feature_names_out(self.categorical_cols).tolist()
            names.extend(cat_names)
        self.feature_names_ = names
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform.")
        transformed = self.pipeline.transform(X)
        return np.asarray(transformed, dtype=np.float32)


def identify_columns(
    df: pd.DataFrame,
    config: WorkstreamConfig,
) -> Tuple[List[str], List[str], List[str]]:
    """Determine excluded, numeric, and categorical columns for model features."""
    all_cols = list(df.columns)
    target = config.target_column

    excluded = set(config.drop_columns or [])
    if config.metadata_columns:
        excluded.update(config.metadata_columns)
    excluded.add(target)
    if "sample_id" in all_cols:
        excluded.add("sample_id")

    candidate_cols = [c for c in all_cols if c not in excluded]

    if config.numeric_columns is not None:
        numeric_cols = [c for c in config.numeric_columns if c in candidate_cols]
    else:
        numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]

    if config.categorical_columns is not None:
        categorical_cols = [c for c in config.categorical_columns if c in candidate_cols]
    else:
        categorical_cols = [c for c in candidate_cols if c not in numeric_cols]

    return list(excluded), numeric_cols, categorical_cols


def prepare_data_and_splits(
    df: pd.DataFrame,
    config: WorkstreamConfig,
    base_dir: Optional[Path] = None,
) -> SplitData:
    """Split data deterministically with leakage guards and fit preprocessors on train only."""
    work_dir = base_dir or Path.cwd()
    reports_dir = config.resolve_path(work_dir, config.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if config.target_column not in df.columns:
        raise ValueError(f"Target column '{config.target_column}' missing from DataFrame.")

    # Clean target
    clean_df = df.dropna(subset=[config.target_column]).copy()
    y_raw = clean_df[config.target_column].astype(str)

    class_counts = y_raw.value_counts()
    if len(class_counts) < 2:
        raise ValueError(f"Dataset has only {len(class_counts)} class; minimum 2 required.")
    if class_counts.min() < 2:
        raise ValueError(f"At least one class has fewer than 2 instances: {class_counts[class_counts < 2].to_dict()}")

    excluded_cols, numeric_cols, categorical_cols = identify_columns(clean_df, config)
    print(f"[PREPROCESSING] Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"[PREPROCESSING] Categorical features ({len(categorical_cols)}): {categorical_cols}")
    print(f"[PREPROCESSING] Excluded columns ({len(excluded_cols)}): {excluded_cols}")

    if not numeric_cols and not categorical_cols:
        raise ValueError("No feature columns remaining after applying exclusion filters.")

    # Train / Test split
    indices = np.arange(len(clean_df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=y_raw,
    )

    train_df = clean_df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = clean_df.iloc[test_idx].copy().reset_index(drop=True)

    # Label encoding
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df[config.target_column].astype(str))

    # Verify all test classes are known
    test_labels = set(test_df[config.target_column].astype(str))
    train_labels = set(label_encoder.classes_)
    unseen_classes = test_labels - train_labels
    if unseen_classes:
        raise ValueError(f"Test split contains unseen classes not present in train: {unseen_classes}")

    y_test = label_encoder.transform(test_df[config.target_column].astype(str))

    # Fit preprocessor strictly on train
    preprocessor = FeaturePreprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    preprocessor.fit(train_df)

    X_train = preprocessor.transform(train_df)
    X_test = preprocessor.transform(test_df)

    # Check for train/test data leakage (exact row overlap on feature matrix)
    # Using hash-based row check
    train_hashes = set(hash(tuple(row)) for row in X_train)
    overlap_count = sum(1 for row in X_test if hash(tuple(row)) in train_hashes)

    # Save split manifest
    split_manifest = {
        "dataset_name": config.dataset_name,
        "random_seed": config.random_seed,
        "test_size": config.test_size,
        "total_rows": len(clean_df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_sample_ids": train_df["sample_id"].tolist() if "sample_id" in train_df else [],
        "test_sample_ids": test_df["sample_id"].tolist() if "sample_id" in test_df else [],
        "train_class_counts": pd.Series(y_train).value_counts().to_dict(),
        "test_class_counts": pd.Series(y_test).value_counts().to_dict(),
        "class_mapping": {str(cls_name): int(idx) for idx, cls_name in enumerate(label_encoder.classes_)},
        "feature_names": preprocessor.feature_names_,
        "feature_count": len(preprocessor.feature_names_),
        "potential_feature_overlap_count": overlap_count,
        "leakage_warning": (
            "Detected exact feature matches between train and test. "
            "May occur in synthetic data or repeated network packets."
            if overlap_count > 0 else "No exact feature overlap detected between train and test."
        ),
    }

    manifest_path = reports_dir / "split_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

    # Export modeling sample for reproducibility
    sample_export_cols = [c for c in clean_df.columns if c in ["sample_id", "flow_id", "timestamp", "source_file", config.target_column] + numeric_cols[:5]]
    clean_df[sample_export_cols].to_csv(reports_dir / "modeling_sample.csv", index=False)

    print(f"[PREPROCESSING] Processed {len(preprocessor.feature_names_)} features across {len(label_encoder.classes_)} classes.")
    print(f"[PREPROCESSING] Train size: {len(X_train)}, Test size: {len(X_test)}. Split manifest saved to {manifest_path}")

    return SplitData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_df=train_df,
        test_df=test_df,
        feature_names=preprocessor.feature_names_,
        class_names=label_encoder.classes_.tolist(),
        preprocessor=preprocessor,
        label_encoder=label_encoder,
        split_manifest=split_manifest,
    )
