from __future__ import annotations
import hashlib
from pathlib import Path
import pandas as pd

NETWORK_COLUMNS = ["ts", "src_ip", "src_port", "dst_ip", "dst_port", "proto", "type"]
IOT_COLUMNS = ["ts", "data", "time", "type"]

def find_csvs(folder: str | Path) -> list[Path]:
    root = Path(folder)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.csv") if p.is_file())

def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def unique_files_by_content(paths: list[Path]) -> tuple[list[Path], list[dict]]:
    """Avoid hashing every large file: hash only files sharing the same byte size."""
    by_size: dict[int, list[Path]] = {}
    for p in paths:
        by_size.setdefault(p.stat().st_size, []).append(p)
    kept, duplicate_rows = [], []
    for size, group in by_size.items():
        if len(group) == 1:
            kept.extend(group)
            continue
        hashes: dict[str, Path] = {}
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

def normalise_label(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
