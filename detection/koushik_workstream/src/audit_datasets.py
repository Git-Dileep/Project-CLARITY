from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from .common import find_csvs, unique_files_by_content, normalise_label

def inspect_file(path: Path, dataset_kind: str, chunksize: int) -> dict:
    rows = 0
    missing = {}
    label_counts: dict[str, int] = {}
    columns = []
    try:
        for i, chunk in enumerate(pd.read_csv(path, chunksize=chunksize, low_memory=False)):
            if i == 0:
                columns = list(chunk.columns)
            rows += len(chunk)
            for col in chunk.columns:
                missing[col] = missing.get(col, 0) + int(chunk[col].isna().sum())
            if "type" in chunk.columns:
                labels = normalise_label(chunk["type"]).dropna().value_counts()
                for label, count in labels.items():
                    label_counts[str(label)] = label_counts.get(str(label), 0) + int(count)
    except Exception as exc:
        return {
            "file": str(path), "dataset_kind": dataset_kind, "status": "ERROR",
            "error": repr(exc), "rows": rows, "columns": "|".join(columns),
            "label_counts_json": json.dumps(label_counts), "missing_json": json.dumps(missing)
        }
    return {
        "file": str(path), "dataset_kind": dataset_kind, "status": "OK",
        "size_bytes": path.stat().st_size, "rows": rows,
        "columns": "|".join(columns), "label_counts_json": json.dumps(label_counts, sort_keys=True),
        "missing_json": json.dumps(missing, sort_keys=True),
        "unique_labels": len(label_counts),
    }

def main():
    parser = argparse.ArgumentParser(description="Chunked audit of network and IoT device CSV datasets.")
    parser.add_argument("--network-dir", default="input_data/network")
    parser.add_argument("--iot-dir", default="input_data/iot_devices")
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()
    out = Path("reports"); out.mkdir(parents=True, exist_ok=True)
    network_files = find_csvs(args.network_dir)
    iot_files = find_csvs(args.iot_dir)
    network_unique, network_dups = unique_files_by_content(network_files)
    iot_unique, iot_dups = unique_files_by_content(iot_files)

    rows = []
    for kind, paths in [("network", network_files), ("iot_device", iot_files)]:
        for path in paths:
            print(f"[AUDIT] {kind}: {path}")
            rows.append(inspect_file(path, kind, args.chunksize))
    audit = pd.DataFrame(rows)
    audit.to_csv(out / "data_audit.csv", index=False)

    dup_records = [{"dataset_kind":"network", **r} for r in network_dups] + [
        {"dataset_kind":"iot_device", **r} for r in iot_dups
    ]
    pd.DataFrame(dup_records, columns=["dataset_kind","duplicate_file","same_as","size_bytes","sha256"]).to_csv(
        out / "duplicate_files.csv", index=False
    )

    summary_counts: dict[str, int] = {}
    for path in network_unique:
        try:
            for chunk in pd.read_csv(path, usecols=["type"], chunksize=args.chunksize, low_memory=False):
                counts = normalise_label(chunk["type"]).dropna().value_counts()
                for label, count in counts.items():
                    summary_counts[str(label)] = summary_counts.get(str(label), 0) + int(count)
        except Exception as exc:
            print(f"[WARN] Cannot count labels in {path}: {exc}")
    pd.DataFrame([{"label": k, "count": v} for k, v in sorted(summary_counts.items(), key=lambda x: x[1], reverse=True)]).to_csv(
        out / "class_summary.csv", index=False
    )

    summary = {
        "network_files_found": len(network_files),
        "network_unique_files_by_content": len(network_unique),
        "network_duplicate_files": len(network_dups),
        "iot_files_found": len(iot_files),
        "iot_unique_files_by_content": len(iot_unique),
        "iot_duplicate_files": len(iot_dups),
        "network_class_counts": summary_counts,
        "network_total_rows": int(audit.loc[(audit.dataset_kind == "network") & (audit.status == "OK"), "rows"].sum()) if not audit.empty else 0,
        "iot_total_rows": int(audit.loc[(audit.dataset_kind == "iot_device") & (audit.status == "OK"), "rows"].sum()) if not audit.empty else 0,
        "notes": [
            "Totals in data_audit.csv include duplicate files; class_summary.csv excludes exact duplicate network files.",
            "The IoT device sample schema ts,data,time,type does not expose independent features besides event metadata.",
            "Audit does not validate official dataset provenance/licence."
        ]
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nAudit complete. Outputs are in reports/.")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
