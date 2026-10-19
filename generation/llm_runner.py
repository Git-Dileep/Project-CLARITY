import os
import json
import copy
from datetime import datetime
from groq import Groq

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

PROMPT_TEMPLATE = """
You are a network security analyst analyzing a generic IoT security alert.
Review the provided Intrusion Detection System (IDS) alert, including its SHAP feature attributions.
Use the retrieved knowledge base evidence (if provided) to explain the attack type, why the system flagged it, and suggest mitigations.

IDS Alert Details:
{alert_data}

Retrieved Knowledge Base Context:
{retrieved_context}

You must respond ONLY with a valid JSON object matching the following structure:
{{
  "incident_summary": "Concise executive overview of the incident",
  "attack_mechanism": "Detailed explanation of technical attack mechanics and protocols",
  "key_indicators": ["indicator 1 based on SHAP", "indicator 2"],
  "recommended_actions": ["action 1", "action 2"],
  "raw_report_markdown": "Full text markdown representation of the report"
}}
"""

def format_retrieval_context(retrieval_data: dict) -> str:
    """Formats retrieved chunks into readable text for the LLM."""
    if not retrieval_data or "chunks" not in retrieval_data or not retrieval_data["chunks"]:
        return "No retrieved context available (Ungrounded Baseline)."
    
    context_str = ""
    for i, chunk in enumerate(retrieval_data["chunks"], 1):
        context_str += f"\n--- Evidence Chunk {i} ---\n"
        context_str += f"Source: {chunk.get('source', 'unknown')}\n"
        context_str += f"Content: {chunk.get('content', '')}\n"
    return context_str

def generate_incident_report(alert_data: dict, retrieval_data: dict = None, system_id: str = "SYS-01") -> dict:
    """Runs the alert and retrieved context through Llama-3.1-8b-instant."""
    
    # 1. Strip evaluation_metadata to comply with Critical Prompt Boundary Rule
    safe_alert_data = copy.deepcopy(alert_data)
    if "evaluation_metadata" in safe_alert_data:
        del safe_alert_data["evaluation_metadata"]
        
    # 2. Format context & build prompt
    formatted_context = format_retrieval_context(retrieval_data)
    prompt = PROMPT_TEMPLATE.format(
        alert_data=json.dumps(safe_alert_data, indent=2), 
        retrieved_context=formatted_context
    )
    
    # 3. Call Llama 3.1 8B Instant
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    llm_output = json.loads(response.choices[0].message.content)
    
    # 4. Construct final payload matching report_schema_v1.json
    final_report = {
        "system_id": system_id,
        "flow_id": alert_data["identity"]["flow_id"],
        "alert_id": alert_data["identity"]["alert_id"],
        "model_name": "meta-llama/Llama-3.1-8b-instant",
        "generation_timestamp": datetime.utcnow().isoformat() + "Z",
        "incident_summary": llm_output.get("incident_summary", ""),
        "attack_mechanism": llm_output.get("attack_mechanism", ""),
        "key_indicators": llm_output.get("key_indicators", []),
        "recommended_actions": llm_output.get("recommended_actions", []),
        "raw_report_markdown": llm_output.get("raw_report_markdown", "")
    }
    
    return final_report

if __name__ == "__main__":
    print("SNE-02 Runner Skeleton initialized successfully.")