import os
import pytest
from retrieval.retrieve import retrieve_for_alert, LOG_FILE

def test_retrieve_for_alert():
    sample_alert = {
        "alert_id": "ALT-2026-001",
        "timestamp": "2026-10-20T10:00:00Z",
        "attack_type": "DDoS-SYN_Flood",
        "predicted_label": "Malicious",
        "confidence": 0.98,
        "top_features": [
            {"feature": "fwd_pkts_per_sec", "importance": 0.42},
            {"feature": "syn_flag_count", "importance": 0.35}
        ],
        "description": "High volume SYN packet rate detected on consumer gateway"
    }

    result = retrieve_for_alert(sample_alert, top_k=3)

    assert result["alert_id"] == "ALT-2026-001"
    assert len(result["chunks"]) <= 3
    assert "query_used" in result
    assert os.path.exists(LOG_FILE)
