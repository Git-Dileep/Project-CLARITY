import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from eval.score_reports import evaluate_claim_llm
except ImportError:
    from score_reports import evaluate_claim_llm
TEST_CASES = [
    {
        "claim": "The unauthorized SSH connection initiated from IP 192.168.1.105 on port 22.",
        "evidence": "Log entry: SSH login attempt from 192.168.1.105:22 flagged as unauthorized anomaly.",
        "expected": "FULL"
    },
    {
        "claim": "Gateway router rebooted cleanly with no dropped packets.",
        "evidence": "Log entry: SSH login attempt from 192.168.1.105:22 flagged as unauthorized anomaly.",
        "expected": "UNSUPPORTED"
    },
    {
        "claim": "The alert was triggered by normal encrypted HTTPS traffic on port 443.",
        "evidence": "Log entry: SSH login attempt from 192.168.1.105:22 flagged as unauthorized anomaly.",
        "expected": "CONTRADICTED"
    },
    {
        "claim": "Anomalous network spike detected on internal gateway interface.",
        "evidence": "Network telemetry shows 500% bandwidth spike on internal gateway interface.",
        "expected": "FULL"
    },
    {
        "claim": "Database server backup completed successfully at 02:00 UTC.",
        "evidence": "Network telemetry shows 500% bandwidth spike on internal gateway interface.",
        "expected": "UNSUPPORTED"
    },
    {
        "claim": "User account admin login failed due to invalid password.",
        "evidence": "Auth log: User admin authenticated successfully at 08:30.",
        "expected": "CONTRADICTED"
    },
    {
        "claim": "Suspicious PowerShell execution detected with encoded payload.",
        "evidence": "EDR alert: powershell.exe executed with -EncodedCommand parameter.",
        "expected": "FULL"
    },
    {
        "claim": "Firewall blocked outbound connection to known C2 IP.",
        "evidence": "Firewall log: DROP TCP 10.0.0.5 -> 185.220.101.5:8080 (Threat ID: C2_BEACON).",
        "expected": "FULL"
    },
    {
        "claim": "DNS query for malware domain was permitted by local resolver.",
        "evidence": "Firewall log: DROP TCP 10.0.0.5 -> 185.220.101.5:8080 (Threat ID: C2_BEACON).",
        "expected": "CONTRADICTED"
    },
    {
        "claim": "System kernel updated to version 6.1.0.",
        "evidence": "Firewall log: DROP TCP 10.0.0.5 -> 185.220.101.5:8080 (Threat ID: C2_BEACON).",
        "expected": "UNSUPPORTED"
    }
]

def main():
    correct = 0
    total = len(TEST_CASES)
    print("Running Judge Validation against Rubric v0 Standard...\n")

    for idx, tc in enumerate(TEST_CASES, 1):
        predicted = evaluate_claim_llm(tc["claim"], tc["evidence"])
        # Support Rubric v0 exact label matching or FULL/SUPPORTED mapping
        is_match = (predicted == tc["expected"]) or (predicted == "SUPPORTED" and tc["expected"] == "FULL")
        if is_match:
            correct += 1
            status = "PASS"
        else:
            status = "FAIL"
        print(f"Test {idx:02d}: [{status}] Expected: {tc['expected']} | Predicted: {predicted}")

    agreement_pct = (correct / total) * 100
    print(f"\nJudge Agreement Score: {correct}/{total} ({agreement_pct:.1f}%)")
    if agreement_pct >= 80.0:
        print("Status: VALIDATED (>= 80% agreement target met)")
    else:
        print("Status: UNVALIDATED (< 80% agreement target)")

if __name__ == "__main__":
    main()