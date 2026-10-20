import os
import json
import logging
from typing import Dict, Any, Optional

# Logging setup
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "tavily")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

TOOLS_LOG = os.path.join(LOG_DIR, "tools.log")

logger = logging.getLogger("retrieval_tools")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(TOOLS_LOG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

# Standard generic-IoT ATT&CK mapping table
ATTACK_MAPPING = {
    "DDoS-SYN_Flood": {
        "technique_id": "T1498.001",
        "technique_name": "Direct Network Flood",
        "tactic": "Impact",
        "url": "https://attack.mitre.org/techniques/T1498/001/"
    },
    "DDoS-UDP_Flood": {
        "technique_id": "T1498.001",
        "technique_name": "Direct Network Flood",
        "tactic": "Impact",
        "url": "https://attack.mitre.org/techniques/T1498/001/"
    },
    "Mirai-BruteForce": {
        "technique_id": "T1110.001",
        "technique_name": "Password Guessing",
        "tactic": "Credential Access",
        "url": "https://attack.mitre.org/techniques/T1110/001/"
    },
    "Recon-PortScan": {
        "technique_id": "T1046",
        "technique_name": "Network Service Discovery",
        "tactic": "Discovery",
        "url": "https://attack.mitre.org/techniques/T1046/"
    },
    "Vulnerability-Exploit": {
        "technique_id": "T1210",
        "technique_name": "Exploitation of Remote Services",
        "tactic": "Lateral Movement",
        "url": "https://attack.mitre.org/techniques/T1210/"
    }
}


def map_mitre_attack(attack_type: str) -> Dict[str, Any]:
    """Map an alert attack type to MITRE ATT&CK technique details."""
    logger.info(f"map_mitre_attack called for attack_type='{attack_type}'")
    mapping = ATTACK_MAPPING.get(attack_type, {
        "technique_id": "T1498",
        "technique_name": "Network Denial of Service",
        "tactic": "Impact",
        "url": "https://attack.mitre.org/techniques/T1498/"
    })
    return mapping


def get_cached_tavily_search(alert_id: str, query: str, mock_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Search Tavily with caching per alert_id.
    If cached, reuses saved results to ensure identical context across detector evaluations.
    """
    cache_file = os.path.join(CACHE_DIR, f"{alert_id}.json")
    
    # 1. Check cache first
    if os.path.exists(cache_file):
        logger.info(f"Tavily cache hit for alert_id='{alert_id}'")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # 2. Perform search (or use fallback mock if API key is unset)
    logger.info(f"Tavily cache miss for alert_id='{alert_id}'. Performing query='{query}'")
    api_key = os.getenv("TAVILY_API_KEY")

    if api_key:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_key)
            results = client.search(query=query, search_depth="basic", max_results=3)
        except Exception as e:
            logger.warning(f"Tavily client failed: {e}. Falling back to default payload.")
            results = {"query": query, "results": mock_results or []}
    else:
        results = {
            "query": query,
            "cached": True,
            "results": mock_results or [
                {
                    "title": f"Threat intel context for {query}",
                    "content": f"Automated IoT telemetry advisory for {query}.",
                    "url": "https://cisa.gov/news-events/alerts"
                }
            ]
        }

    # 3. Save to cache
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results
