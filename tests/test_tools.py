import os
import pytest
from retrieval.tools import map_mitre_attack, get_cached_tavily_search, TOOLS_LOG, CACHE_DIR

def test_mitre_attack_mapping():
    result = map_mitre_attack("DDoS-SYN_Flood")
    assert result["technique_id"] == "T1498.001"
    assert "Direct Network Flood" in result["technique_name"]
    assert os.path.exists(TOOLS_LOG)

def test_tavily_caching():
    alert_id = "TEST-ALERT-001"
    cache_path = os.path.join(CACHE_DIR, f"{alert_id}.json")
    if os.path.exists(cache_path):
        os.remove(cache_path)

    # First call: cache miss
    res1 = get_cached_tavily_search(alert_id, "SYN Flood IoT attack")
    assert os.path.exists(cache_path)

    # Second call: cache hit
    res2 = get_cached_tavily_search(alert_id, "SYN Flood IoT attack")
    assert res1 == res2
