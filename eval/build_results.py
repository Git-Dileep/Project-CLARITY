import os
import json

RESULTS_MD = os.path.join("outputs", "eval_results.md")
REFERENCE_METRICS = {
    "Traditional ML ensemble": os.path.join(
        "detection", "workstream", "reports", "aci_iot_2023", "traditional_metrics.json"
    ),
    "Deep learning ensemble": os.path.join(
        "detection", "workstream", "reports", "aci_iot_2023", "dl", "dl_metrics.json"
    ),
}

def run_evaluation_and_build_tables():
    rows = []
    for model_name, metrics_path in REFERENCE_METRICS.items():
        with open(metrics_path, "r", encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)["ensemble"]

        test_rows = int(metrics["classification_report"]["macro avg"]["support"])
        rows.append(
            f"| {model_name} | {metrics['accuracy']:.1%} | "
            f"{metrics['macro_f1']:.4f} | {test_rows:,} |"
        )

    full_md = "\n".join(
        [
            "# Evaluation Results: ACI IoT Network Dataset 2023",
            "",
            "> **Final study results evaluated on the real-world ACI IoT Network Dataset 2023.**",
            "",
            "The ACI IoT 2023 dataset contains 1,231,411 rows across 12 attack/traffic classes. "
            "The table reports detector ensemble performance on the 17,757-row held-out test split.",
            "",
            "| Detector | Accuracy | Macro F1 | Test rows |",
            "|:---|---:|---:|---:|",
            *rows,
            "",
            "A separate blind review of five generated pilot reports is documented in "
            "`evaluation/human_eval.md`.",
            "",
        ]
    )

    os.makedirs(os.path.dirname(RESULTS_MD), exist_ok=True)
    with open(RESULTS_MD, "w", encoding="utf-8") as f:
        f.write(full_md)
        
    print(full_md)
    print(f"Results successfully written to {RESULTS_MD}")

if __name__ == "__main__":
    run_evaluation_and_build_tables()