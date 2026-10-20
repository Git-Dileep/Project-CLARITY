import os
import json
from groq import Groq

def generate_incident_report(alert_data, retrieval_data=None, system_id="SYS-01"):
    """
    Generates a schema-compliant incident report using Groq API.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")

    client = Groq(api_key=api_key)

    prompt = f"""You are an expert IoT cybersecurity incident reporter.
Analyze the following alert payload and generate a concise incident report.

ALERT DATA:
{json.dumps(alert_data, indent=2)}

RETRIEVAL DATA:
{json.dumps(retrieval_data, indent=2) if retrieval_data else "None"}

Instructions:
- Return ONLY a valid JSON object.
- Keep text explanations concise and direct to avoid token limits.
- The JSON object must include:
  "incident_summary": brief summary of the attack,
  "severity_assessment": severity level (e.g., HIGH, MEDIUM, LOW),
  "top_risk_factors": list of top feature drivers from SHAP,
  "mitigation_recommendations": list of 2-3 specific action steps.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=4096,
        response_format={"type": "json_object"}
    )

    llm_output = json.loads(response.choices[0].message.content)

    final_report = {
        "system_id": system_id,
        "flow_id": alert_data.get("identity", {}).get("flow_id", "UNKNOWN"),
        "alert_id": alert_data.get("identity", {}).get("alert_id", "UNKNOWN"),
        "generated_report": llm_output
    }

    return final_report