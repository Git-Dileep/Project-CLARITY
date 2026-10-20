import sys
import os

# Add root folder to sys.path so 'from eval...' works cleanly from anywhere
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.score_reports import evaluate_claim_llm

HAND_LABELLED_CLAIMS = [
    {"evidence": "IP 10.0.0.1 initiated port scan on 10.0.0.5.", "claim": "Port scan originated from 10.0.0.1.", "human_label": "SUPPORTED"},
    {"evidence": "Device A CPU utilization reached 98%.", "claim": "Device A CPU spiked to 98%.", "human_label": "SUPPORTED"},
    {"evidence": "User admin logged in at 14:00 UTC.", "claim": "User admin logged in at 14:00 UTC.", "human_label": "SUPPORTED"},
    {"evidence": "Malware payload identified as Trojan.Generic.", "claim": "Payload confirmed as Ransomware.LockBit.", "human_label": "CONTRADICTED"},
    {"evidence": "Database connection timed out after 30 seconds.", "claim": "Database server was physically destroyed.", "human_label": "UNGROUNDED"},
    {"evidence": "MQTT message burst of 5000 msg/sec detected.", "claim": "MQTT broker received high volume message burst.", "human_label": "SUPPORTED"},
    {"evidence": "Firmware revision updated to v2.4.1.", "claim": "Firmware downgraded to v1.0.0.", "human_label": "CONTRADICTED"},
    {"evidence": "API endpoint returned HTTP status 500.", "claim": "API returned 500 Internal Server Error.", "human_label": "SUPPORTED"},
    {"evidence": "DNS lookup failed for domain malware-c2.com.", "claim": "DNS request succeeded without issues.", "human_label": "CONTRADICTED"},
    {"evidence": "Sensor node 12 battery level at 15%.", "claim": "Sensor node 12 battery is low.", "human_label": "SUPPORTED"}
]

def run_validation():
    print("=== SNE-02 LLM Judge vs Human Hand-Labelled Claims Validation ===\n")
    matches = 0
    results = []

    for idx, item in enumerate(HAND_LABELLED_CLAIMS, 1):
        llm_label = evaluate_claim_llm(item["claim"], item["evidence"])
        is_match = (llm_label == item["human_label"])
        if is_match:
            matches += 1
        results.append((idx, item["claim"], item["human_label"], llm_label, "✅ Match" if is_match else "❌ Mismatch"))

    print(f"| Claim ID | Claim Text | Human Label | LLM Judge Label | Status |")
    print(f"|---|---|---|---|---|")
    for r in results:
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")

    agreement = (matches / len(HAND_LABELLED_CLAIMS)) * 100
    print(f"\nAgreement Score: {matches}/{len(HAND_LABELLED_CLAIMS)} ({agreement:.1f}%)")
    print("Acceptance Criteria Met: " + ("YES (>= 80%)" if matches >= 8 else "NO"))

if __name__ == "__main__":
    run_validation()