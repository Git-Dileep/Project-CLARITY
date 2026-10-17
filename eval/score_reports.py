"""
Evaluation Engine for LLM Incident Reports (Rubric v0).

Computes Groundedness, Faithfulness, Fabrication Rate, and Hallucination Rate
across extracted atomic claims according to DIL-02 specification.
"""

from typing import Literal, Optional
import json
from pydantic import BaseModel, Field

LabelType = Literal["FULL", "PARTIAL", "UNSUPPORTED", "CONTRADICTED", "FABRICATED"]

class AtomicClaim(BaseModel):
    claim_id: int
    text: str
    label: LabelType
    justification: str

class MetricScores(BaseModel):
    total_claims: int
    full_count: int
    partial_count: int
    unsupported_count: int
    contradicted_count: int
    fabricated_count: int
    groundedness: float = Field(ge=0.0, le=1.0)
    faithfulness: float = Field(ge=0.0, le=1.0)
    fabrication_rate: float = Field(ge=0.0, le=1.0)
    hallucination_rate: float = Field(ge=0.0, le=1.0)

class ReportEvaluation(BaseModel):
    claims: list[AtomicClaim]
    metrics: MetricScores

def apply_tie_break(
    is_fabricated: bool,
    is_contradicted: bool,
    is_unsupported: bool,
    is_full: bool
) -> LabelType:
    """
    Enforces the strict priority tie-break sequence:
    FABRICATED -> CONTRADICTED -> UNSUPPORTED -> (FULL if is_full else PARTIAL).
    """
    if is_fabricated:
        return "FABRICATED"
    if is_contradicted:
        return "CONTRADICTED"
    if is_unsupported:
        return "UNSUPPORTED"
    if is_full:
        return "FULL"
    return "PARTIAL"

def calculate_metrics(claims: list[AtomicClaim]) -> MetricScores:
    """
    Calculates metric ratios from atomic claims according to Rubric v0 formulas:
      - Groundedness = (FULL + PARTIAL) / N
      - Faithfulness = FULL / N
      - Fabrication Rate = FABRICATED / N
      - Hallucination Rate = (CONTRADICTED + UNSUPPORTED) / N

    Provides safe division handling when total_claims == 0.
    """
    total = len(claims)
    if total == 0:
        return MetricScores(
            total_claims=0,
            full_count=0,
            partial_count=0,
            unsupported_count=0,
            contradicted_count=0,
            fabricated_count=0,
            groundedness=0.0,
            faithfulness=0.0,
            fabrication_rate=0.0,
            hallucination_rate=0.0,
        )

    full = sum(1 for c in claims if c.label == "FULL")
    partial = sum(1 for c in claims if c.label == "PARTIAL")
    unsupported = sum(1 for c in claims if c.label == "UNSUPPORTED")
    contradicted = sum(1 for c in claims if c.label == "CONTRADICTED")
    fabricated = sum(1 for c in claims if c.label == "FABRICATED")

    groundedness = (full + partial) / total
    faithfulness = full / total
    fabrication_rate = fabricated / total
    hallucination_rate = (contradicted + unsupported) / total

    # Floating point sanity verification
    sum_identity = faithfulness + (partial / total) + fabrication_rate + hallucination_rate
    assert abs(sum_identity - 1.0) < 1e-6, f"Completeness identity violated: {sum_identity}"

    return MetricScores(
        total_claims=total,
        full_count=full,
        partial_count=partial,
        unsupported_count=unsupported,
        contradicted_count=contradicted,
        fabricated_count=fabricated,
        groundedness=round(groundedness, 4),
        faithfulness=round(faithfulness, 4),
        fabrication_rate=round(fabrication_rate, 4),
        hallucination_rate=round(hallucination_rate, 4),
    )

def evaluate_report(
    evidence_text: str,
    report_text: str,
    client: Optional[object] = None,
    model: str = "meta-llama/Llama-3.1-8b-instant"
) -> ReportEvaluation:
    """
    Extracts atomic claims and evaluates them against evidence_text.
    If client is None or in offline test mode, returns a calibrated mock evaluation
    covering all 5 labels.
    """
    if client is None:
        # Calibrated mock evaluation for testing and offline integration
        mock_claims = [
            AtomicClaim(
                claim_id=1,
                text="The attack was classified as DDoS SYN Flood with 94.2% confidence.",
                label="FULL",
                justification="Direct match with prediction metadata in E.",
            ),
            AtomicClaim(
                claim_id=2,
                text="Observed SYN flag count reached 18,450 packets.",
                label="FULL",
                justification="Direct numerical match with rank 1 top_features.",
            ),
            AtomicClaim(
                claim_id=3,
                text="Traffic demonstrated pronounced asymmetry between inbound and outbound volume.",
                label="PARTIAL",
                justification="Conceptually accurate regarding dst_to_src_byte_ratio, but omits specific numerical values.",
            ),
            AtomicClaim(
                claim_id=4,
                text="The compromised device is likely part of the Mirai IoT botnet ecosystem.",
                label="UNSUPPORTED",
                justification="Speculative deduction not found in E, though no specific verifiable token was invented.",
            ),
            AtomicClaim(
                claim_id=5,
                text="The automated detector exhibited low confidence in classifying the threat.",
                label="CONTRADICTED",
                justification="Contradicts the 0.942 detector confidence reported in E.",
            ),
            AtomicClaim(
                claim_id=6,
                text="Inspection confirmed exploitation of CVE-2023-1389 on the target device.",
                label="FABRICATED",
                justification="Invented specific CVE identifier not present in E.",
            ),
        ]
        metrics = calculate_metrics(mock_claims)
        return ReportEvaluation(claims=mock_claims, metrics=metrics)

    # Real client invocation stub (e.g. Groq, OpenAI, or local vLLM API)
    # The prompt adheres to the Judge LLM Prompt Specification in eval/rubric_v0.md
    prompt = (
        f"You are an expert cybersecurity evaluation auditor grading LLM incident reports.\n"
        f"REFERENCE EVIDENCE SET (E):\n{evidence_text}\n\n"
        f"INCIDENT REPORT:\n{report_text}\n\n"
        f"Extract all atomic claims and assign exactly one label per claim: "
        f"'FULL', 'PARTIAL', 'UNSUPPORTED', 'CONTRADICTED', 'FABRICATED'.\n"
        f"Output JSON matching: {{\"claims\": [{{\"claim_id\": 1, \"text\": \"...\", \"label\": \"...\", \"justification\": \"...\"}}]}}"
    )
    # Generic client call handling
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    raw_json = json.loads(response.choices[0].message.content)
    claims = [AtomicClaim(**c) for c in raw_json.get("claims", [])]
    metrics = calculate_metrics(claims)
    return ReportEvaluation(claims=claims, metrics=metrics)
