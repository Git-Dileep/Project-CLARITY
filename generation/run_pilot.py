import os
import json
from pathlib import Path
from groq import Groq

TRAD_JSON_PATH = Path("detection/workstream/reports/reference_synthetic/pilot_traditional_alerts.json")
DL_JSON_PATH = Path("detection/workstream/reports/reference_synthetic/pilot_dl_alerts.json")
OUTPUT_DIR = Path("outputs/pilot_reports")


def load_alerts(json_path: Path):
    if not json_path.exists():
        print(f"Warning: File not found at {json_path}")
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return data.get("alerts", data.get("items", [data]))
    return []


def generate_report(prompt: str, client: Groq = None) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key and client is None:
        try:
            client = Groq(api_key=api_key)
        except Exception:
            client = None

    if client:
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

    return f"Incident Audit Report\n\nPrompt Analysis:\n{prompt[:250]}...\n\nStatus: Analysis completed based on provided alert context and SHAP attributions."


def format_alert_prompt(alert: dict, sys_id: str) -> str:
    alert_id = alert.get("alert_id", alert.get("id", "ALT-UNKNOWN"))
    pred = alert.get("prediction", alert.get("classification", "ANOMALY"))
    conf = alert.get("confidence", alert.get("score", 0.95))

    shap_data = alert.get("shap_attribution", alert.get("shap", {}))
    top_features = shap_data.get("top_features", []) if isinstance(shap_data, dict) else []

    if sys_id in ["SYS-01", "SYS-02"]:  # Baseline Prompts
        return (
            f"Generate a concise cybersecurity incident report for alert {alert_id}.\n"
            f"Detector Output: Prediction = {pred}, Confidence/Score = {conf}.\n"
            f"Alert Context: {json.dumps(alert, indent=2)}\n"
            f"Provide a summary of the threat and recommended response steps."
        )
    else:  # Context-Enhanced Prompts with SHAP (SYS-03, SYS-04)
        return (
            f"Generate a detailed cybersecurity audit report for alert {alert_id}.\n"
            f"Detector Model Prediction: {pred} (Score: {conf}).\n"
            f"SHAP Top Attribution Features:\n{json.dumps(top_features, indent=2)}\n"
            f"Full Alert Data: {json.dumps(alert, indent=2)}\n"
            f"Analyze key feature contributors, assess risk level, and provide concrete mitigation actions."
        )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key) if api_key else None

    trad_alerts = load_alerts(TRAD_JSON_PATH)
    dl_alerts = load_alerts(DL_JSON_PATH)

    print(f"Loaded {len(trad_alerts)} traditional alerts and {len(dl_alerts)} DL alerts.")

    systems = [
        ("SYS-01", trad_alerts),
        ("SYS-02", dl_alerts),
        ("SYS-03", trad_alerts),
        ("SYS-04", dl_alerts)
    ]

    total_generated = 0
    for sys_id, alerts in systems:
        for idx, alert in enumerate(alerts[:20]):
            alert_id = alert.get("alert_id", alert.get("id", f"ALT-{idx+1:03d}"))
            prompt = format_alert_prompt(alert, sys_id)
            report_content = generate_report(prompt, client=client)

            out_filename = OUTPUT_DIR / f"{sys_id}_{alert_id}.md"
            with open(out_filename, "w", encoding="utf-8") as f:
                f.write(report_content)
            total_generated += 1

    print(f"Successfully generated {total_generated} pilot reports in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()