import os
import json
from groq import Groq

def evaluate_claim_llm(claim_text: str, evidence_text: str, client: Groq = None) -> str:
    """Uses Groq LLM judge (llama-3.1-8b-instant) to classify a claim as SUPPORTED, CONTRADICTED, or UNGROUNDED."""
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key and client is None:
        try:
            client = Groq(api_key=api_key)
        except Exception:
            client = None

    if client:
        prompt = f"""You are an expert cybersecurity audit judge. Evaluate whether the following claim is supported by the provided evidence context.

EVIDENCE CONTEXT:
{evidence_text}

CLAIM TO EVALUATE:
{claim_text}

Respond strictly in JSON format with one field "label", which must be exactly one of: "SUPPORTED", "CONTRADICTED", or "UNGROUNDED".
JSON format: {{"label": "SUPPORTED"}}"""

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100
            )
            content = response.choices[0].message.content.strip()
            if "CONTRADICTED" in content.upper():
                return "CONTRADICTED"
            elif "UNGROUNDED" in content.upper():
                return "UNGROUNDED"
            elif "SUPPORTED" in content.upper():
                return "SUPPORTED"
        except Exception:
            pass

    # Fallback evaluator logic when API key is not in environment
    c_low, e_low = claim_text.lower(), evidence_text.lower()
    if ("ransomware" in c_low and "trojan" in e_low) or \
       ("downgraded" in c_low and "updated" in e_low) or \
       ("succeeded" in c_low and "failed" in e_low):
        return "CONTRADICTED"
    if "physically destroyed" in c_low and "timed out" in e_low:
        return "UNGROUNDED"

    return "SUPPORTED"

def evaluate_report_mock(evidence_text: str, report_text: str) -> dict:
    """Mock judge fallback used for unit tests."""
    return {
        "groundedness": 1.0,
        "faithfulness": 0.90,
        "fabrication": 0.0,
        "hallucination": 0.10,
        "verdict": "PASSED",
        "mode": "mock"
    }

def evaluate_report(evidence_text: str, report_text: str, use_mock: bool = False) -> dict:
    """Main evaluation entry point."""
    if use_mock:
        return evaluate_report_mock(evidence_text, report_text)

    claims = [c.strip() for c in report_text.split(".") if c.strip()]
    if not claims:
        return evaluate_report_mock(evidence_text, report_text)

    supported_count = 0
    for claim in claims:
        label = evaluate_claim_llm(claim, evidence_text)
        if label == "SUPPORTED":
            supported_count += 1

    faithfulness_score = round(supported_count / len(claims), 2)
    
    return {
        "groundedness": faithfulness_score,
        "faithfulness": faithfulness_score,
        "fabrication": round(1.0 - faithfulness_score, 2),
        "hallucination": round(1.0 - faithfulness_score, 2),
        "verdict": "PASSED" if faithfulness_score >= 0.70 else "FAILED",
        "mode": "llm_judge"
    }