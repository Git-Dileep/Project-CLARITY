"""Dataset loader supporting single/multiple CSVs, glob patterns, schema compatibility checks, and bounded sampling.
"""

from __future__ import annotations

import glob
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

from .config import WorkstreamConfig
from .synthetic_data import save_synthetic_data


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def find_files(pattern_or_path: str | Path, base_dir: Optional[Path] = None) -> List[Path]:
    """Find files matching a file path, directory, or glob pattern."""
    p_str = str(pattern_or_path)
    if base_dir and not Path(p_str).is_absolute():
        resolved_str = str(base_dir / p_str)
    else:
        resolved_str = p_str

    matched = glob.glob(resolved_str, recursive=True)
    paths = [Path(p) for p in matched if Path(p).is_file() and p.endswith(".csv")]
    if not paths and Path(resolved_str).is_dir():
        paths = sorted(Path(resolved_str).rglob("*.csv"))
    return sorted(paths)


def unique_files_by_content(paths: List[Path]) -> Tuple[List[Path], List[Dict[str, Any]]]:
    """Deduplicate files with identical size and sha256 checksum."""
    by_size: Dict[int, List[Path]] = {}
    for p in paths:
        try:
            by_size.setdefault(p.stat().st_size, []).append(p)
        except OSError:
            continue

    kept: List[Path] = []
    duplicate_rows: List[Dict[str, Any]] = []

    for size, group in by_size.items():
        if len(group) == 1:
            kept.extend(group)
            continue
        hashes: Dict[str, Path] = {}
        for p in group:
            digest = sha256_file(p)
            if digest in hashes:
                duplicate_rows.append({
                    "duplicate_file": str(p),
                    "same_as": str(hashes[digest]),
                    "size_bytes": size,
                    "sha256": digest,
                })
            else:
                hashes[digest] = p
                kept.append(p)
    return sorted(kept), duplicate_rows


def check_schema_compatibility(
    files: List[Path],
    target_col: str,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Check if all CSV files share compatible columns and include target_col."""
    if not files:
        return False, [], {"error": "No files provided"}

    file_headers: Dict[str, List[str]] = {}
    for f in files:
        try:
            sample_df = pd.read_csv(f, nrows=2)
            file_headers[str(f)] = list(sample_df.columns)
        except Exception as e:
            return False, [], {"error": f"Failed to read header of {f}: {e}"}

    first_file = files[0]
    canonical_cols = set(file_headers[str(first_file)])
    mismatches: Dict[str, Dict[str, Any]] = {}

    for f in files:
        cols = set(file_headers[str(f)])
        missing = canonical_cols - cols
        extra = cols - canonical_cols
        has_target = target_col in cols
        if missing or extra or not has_target:
            mismatches[str(f)] = {
                "missing_columns": list(missing),
                "extra_columns": list(extra),
                "has_target": has_target,
            }

    is_compatible = len(mismatches) == 0
    common_cols = sorted(list(canonical_cols))
    return is_compatible, common_cols, {
        "canonical_columns": common_cols,
        "mismatched_files": mismatches,
        "target_column": target_col,
    }


def normalize_label(s: pd.Series) -> pd.Series:
    """Clean and strip target labels, standardizing empty representations."""
    return s.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "null": pd.NA})


def load_dataset(
    config: WorkstreamConfig,
    base_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load dataset based on configuration with schema checks, sampling, and leakage guards."""
    resolved_input = config.resolve_path(base_dir or Path.cwd(), config.input_path)
    all_files = find_files(resolved_input, base_dir)

    # Auto-generate synthetic reference if in synthetic mode and file is not yet created
    if config.is_synthetic and not all_files:
        print(f"[DATA_LOADER] Synthetic mode enabled and no data found. Generating synthetic reference dataset...")
        save_synthetic_data(resolved_input, n_rows=5000, seed=config.random_seed, target_col=config.target_column)
        all_files = find_files(resolved_input, base_dir)

    if not all_files:
        raise FileNotFoundError(
            f"No CSV dataset files found matching path/pattern: {config.input_path}. "
            f"If auditing ACI 2023, place dataset in {config.input_path} or configure a valid path."
        )

    unique_files, duplicates = unique_files_by_content(all_files)
    is_compatible, common_cols, schema_report = check_schema_compatibility(unique_files, config.target_column)

    if not is_compatible:
        mismatched_str = "\n".join([f"  {k}: {v}" for k, v in schema_report["mismatched_files"].items()])
        raise ValueError(
            f"Schema compatibility error: CSV files do not have identical columns or target column '{config.target_column}' is missing:\n"
            f"{mismatched_str}"
        )

    if config.target_column not in common_cols:
        raise ValueError(
            f"Configured target column '{config.target_column}' not found in dataset columns: {common_cols}"
        )

    pieces: List[pd.DataFrame] = []
    chunk_quota = config.max_rows_per_class // 10 if config.max_rows_per_class else None

    for path in unique_files:
        print(f"[DATA_LOADER] Loading {path.name}...")
        try:
            reader = pd.read_csv(path, chunksize=config.chunksize, low_memory=False)
            for chunk in reader:
                chunk["source_file"] = path.name
                chunk[config.target_column] = normalize_label(chunk[config.target_column])
                chunk = chunk.dropna(subset=[config.target_column])
                if config.max_rows_per_class and chunk_quota:
                    sampled = []
                    for _, group in chunk.groupby(config.target_column, sort=False):
                        n = min(len(group), chunk_quota)
                        sampled.append(group.sample(n=n, random_state=config.random_seed) if len(group) > n else group)
                    if sampled:
                        pieces.append(pd.concat(sampled, ignore_index=True))
                else:
                    pieces.append(chunk)
        except Exception as exc:
            raise RuntimeError(f"Error reading file {path}: {exc}") from exc

    if not pieces:
        raise RuntimeError("No usable data rows could be loaded from input files.")

    data = pd.concat(pieces, ignore_index=True)
    data = data.drop_duplicates()

    if config.max_rows_per_class:
        final_pieces = []
        for label, group in data.groupby(config.target_column, sort=False):
            n = min(len(group), config.max_rows_per_class)
            final_pieces.append(group.sample(n=n, random_state=config.random_seed) if len(group) > n else group)
        data = pd.concat(final_pieces, ignore_index=True)

    # Deterministic shuffle
    data = data.sample(frac=1.0, random_state=config.random_seed).reset_index(drop=True)
    if "sample_id" not in data.columns:
        data.insert(0, "sample_id", np.arange(len(data), dtype=np.int64))

    metadata = {
        "total_files_found": len(all_files),
        "unique_files_used": len(unique_files),
        "duplicate_files_skipped": len(duplicates),
        "total_rows_loaded": len(data),
        "class_counts": data[config.target_column].value_counts().to_dict(),
        "columns": list(data.columns),
        "duplicates": duplicates,
    }
    return data, metadata
