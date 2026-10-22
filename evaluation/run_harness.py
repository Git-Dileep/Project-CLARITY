import json
import os
from eval.score_reports import evaluate_report

def evaluate_generated_reports(reports_file_path: str, output_results_path: str):
    """Loads generated incident reports and computes Rubric v0 scores."""
    if not os.path.exists(reports_file_path):
        print(f"Error: {reports_file_path} not found.")
        return

    with open(reports_file_path, 'r') as f:
        reports = json.load(f)

    evaluated_results = []
    
    for report in reports:
        metrics = calculate_metrics(report)
        evaluated_results.append({
            "alert_id": report.get("alert_id"),
            "system_id": report.get("system_id"),
            "metrics": metrics
        })

    with open(output_results_path, 'w') as f:
        json.dump(evaluated_results, f, indent=2)

    print(f"Evaluated {len(evaluated_results)} reports -> Saved to {output_results_path}")

if __name__ == "__main__":
    print("GEN-04 Evaluation Harness Skeleton initialized successfully.")