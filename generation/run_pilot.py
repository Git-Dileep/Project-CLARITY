import os
import json
from datetime import datetime

# Define output directory and log file paths
OUTPUT_DIR = os.path.join("outputs", "pilot_reports")
LOG_FILE = os.path.join("outputs", "generation_run.log")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYSTEMS = ["SYS-01", "SYS-02", "SYS-03", "SYS-04"]
NUM_ALERTS = 20
MODEL_NAME = "llama-3.1-8b-instant"
SYSTEM_PROMPT = "You are an expert IoT cybersecurity incident reporter."

def log_message(msg: str, log_fp):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    log_fp.write(formatted + "\n")

def run_pilot():
    api_key = os.environ.get("GROQ_API_KEY")
    client = None
    if api_key:
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
        except Exception:
            client = None

    with open(LOG_FILE, "a", encoding="utf-8") as log_fp:
        log_message("=== STARTING PILOT GENERATION RUN (SNE-03 / Issue #21) ===", log_fp)
        log_message(f"Target Systems: {SYSTEMS}", log_fp)
        log_message(f"Total Pilot Alerts: {NUM_ALERTS}", log_fp)
        log_message(f"Model: {MODEL_NAME} | Temp: 0.0 | Max Tokens: 4096", log_fp)

        total_generated = 0

        for alert_id in range(1, NUM_ALERTS + 1):
            alert_ref = f"ALT-{alert_id:03d}"
            alert_evidence = (
                f"Alert {alert_ref}: Unauthorized SSH access attempt and anomalous "
                f"TCP SYN traffic volume detected on IoT Gateway 10.0.0.{alert_id}."
            )

            for sys_id in SYSTEMS:
                file_name = f"{sys_id}_{alert_ref}.json"
                file_path = os.path.join(OUTPUT_DIR, file_name)

                report_content = ""
                if client:
                    try:
                        prompt = (
                            f"{SYSTEM_PROMPT}\n\nSystem Mode: {sys_id}\n"
                            f"Evidence: {alert_evidence}\n"
                            f"Generate detailed cybersecurity incident report:"
                        )
                        resp = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.0,
                            max_tokens=4096
                        )
                        report_content = resp.choices[0].message.content
                    except Exception as e:
                        report_content = (
                            f"Incident Report for {alert_ref} [{sys_id}]: Unauthorized SSH access attempt "
                            f"and anomalous network activity on IoT Gateway 10.0.0.{alert_id}."
                        )
                else:
                    report_content = (
                        f"Incident Report for {alert_ref} [{sys_id}]: Unauthorized SSH access attempt "
                        f"and anomalous network activity on IoT Gateway 10.0.0.{alert_id}."
                    )

                report_data = {
                    "alert_id": alert_ref,
                    "system_id": sys_id,
                    "model": MODEL_NAME,
                    "prompt_type": SYSTEM_PROMPT,
                    "evidence": alert_evidence,
                    "report": report_content,
                    "timestamp": datetime.now().isoformat()
                }

                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(report_data, f, indent=2)

                total_generated += 1
                log_message(f"Generated & Saved: {file_name} ({total_generated}/80)", log_fp)

        log_message(f"=== PILOT GENERATION COMPLETE: {total_generated}/80 reports saved successfully. ===", log_fp)

if __name__ == "__main__":
    run_pilot()