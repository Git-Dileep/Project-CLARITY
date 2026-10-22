# Evaluation Rubric v0: Reliability Scoring for LLM Incident Reports

**Status:** DRAFT / FROZEN
**Date:** October 17, 2026  
**Document ID:** RUBRIC-V0
**Target Architecture:** RAG Incident Analysis Pipeline (SYS-01 through SYS-06)

---

## 1. Unit of Analysis & Reference Evidence Set ($E$)

### 1.1 Atomic Claim Extraction
The fundamental unit of evaluation is the **Atomic Claim** $c_i \in C$. An atomic claim is a single, self-contained factual assertion extracted from the LLM-generated incident report that cannot be decomposed further without losing its core propositional meaning.

Each incident report $R$ is decomposed into a set of $N$ disjoint claims:
$$R = \{c_1, c_2, \dots, c_N\}$$

### 1.2 Reference Evidence Set ($E$)
Each claim $c_i$ is evaluated strictly and exclusively against the **Reference Evidence Set ($E$)**, defined as:
$$E = E_{\text{alert}} \cup E_{\text{shap}} \cup E_{\text{retrieved\_kb}} \cup E_{\text{tavily\_cache}}$$

1. **$E_{\text{alert}}$ (Alert Payload Context):** The alert JSON instance passed to the RAG generator, **strictly omitting** the `evaluation_metadata` section (to avoid evaluation leakage). Includes `identity` metadata and `prediction` outputs.
2. **$E_{\text{shap}}$ (Top-10 Feature Attributions):** The exactly 10 ranked features, their observed flow values, and signed SHAP attributions from `shap_attribution.top_features`.
3. **$E_{\text{retrieved\_kb}}$ (Retrieved Knowledge Chunks):** Top-$k$ ($k=5$) retrieved dense vector chunks from the local ChromaDB store (MITRE ATT&CK techniques, network protocol documentation, IoT device profiling).
4. **$E_{\text{tavily\_cache}}$ (Retrieved Web Context):** External search results cached for the specific incident query (when evaluating hybrid systems SYS-05 and SYS-06).

> [!IMPORTANT]
> **Closed-World Assumption:** World knowledge not present in $E$ is considered **unverified** for the specific flow incident. Claims asserting specific operational facts about the target network or incident that do not appear in $E$ must not be marked as `FULL`.

---

## 2. Categorical Labels & Strict Definitions

Every atomic claim $c_i$ must be assigned exactly one of the following five mutually exclusive categorical labels:

| Label | Definition | Verification Criteria |
| :--- | :--- | :--- |
| **`FABRICATED`** | Asserts an invented, checkable specific item absent from $E$. | The claim explicitly names an invented CVE ID (e.g., CVE-2023-XXXX), external IP/FQDN, MITRE technique ID (e.g., T1071.001), exact quantitative packet metric, or network feature name **not present** anywhere in $E$. |
| **`CONTRADICTED`** | Directly conflicts with or negates explicit facts in $E$. | The claim asserts values or conclusions that oppose $E$ (e.g., stating the attack was "low volume" when $E_{\text{shap}}$ records `packet_rate_per_sec = 3120.4`, or calling it "benign" when consensus prediction was `DDoS_SYN_Flood`). |
| **`UNSUPPORTED`** | Speculative or uncorroborated qualitative deduction without invented specific tokens. | The claim introduces incident-specific deductions or generic botnet/malware architectural trivia not corroborated by $E$, but does not invent specific verifiable identifiers or numerical metrics. |
| **`PARTIAL`** | Supported in spirit or concept, but over-generalized, imprecise, or hedged. | The claim aligns generally with facts in $E$ but omits key constraints, uses overly broad characterizations, or merges disparate facts imprecisely. |
| **`FULL`** | 100% corroborated by explicit facts in $E$. | Every predicate, entity, and metric within the atomic claim is fully substantiated by explicit textual or numerical data in $E$. |

---

## 3. Strict Priority Tie-Break Sequence

When a claim exhibits characteristics spanning multiple definitions (e.g., a claim that is both uncorroborated and contains an invented CVE number), evaluators and Judge models **must** apply the following strict precedence order:

$$\mathbf{FABRICATED} \succ \mathbf{CONTRADICTED} \succ \mathbf{UNSUPPORTED} \succ \mathbf{FULL} \mid \mathbf{PARTIAL}$$

1. **Rule 1 (`FABRICATED` Precedence):** If the claim asserts any specific checkable identifier, metric, or feature name absent from $E$, it is **`FABRICATED`**, regardless of whether the broader deduction is plausible or partially supported.
2. **Rule 2 (`CONTRADICTED` Precedence):** If the claim conflicts with a direct factual datum in $E$ and contains no fabricated identifiers, it is **`CONTRADICTED`**.
3. **Rule 3 (`UNSUPPORTED` Precedence):** If the claim contains claims outside $E$ without direct contradiction or invented checkable entities, it is **`UNSUPPORTED`**.
4. **Rule 4 (`FULL` vs `PARTIAL` Resolution):** If corroborated, mark **`FULL`** only if strictly and fully corroborated; otherwise, mark **`PARTIAL`**.

---

## 4. Mathematical Metric Formulations

Given a report $R$ with $N$ atomic claims, let $N_{\text{FULL}}$, $N_{\text{PARTIAL}}$, $N_{\text{UNSUPPORTED}}$, $N_{\text{CONTRADICTED}}$, and $N_{\text{FABRICATED}}$ denote the label counts such that:
$$N = N_{\text{FULL}} + N_{\text{PARTIAL}} + N_{\text{UNSUPPORTED}} + N_{\text{CONTRADICTED}} + N_{\text{FABRICATED}}$$

The evaluation metrics are computed as follows:

$$\text{Groundedness} = \frac{N_{\text{FULL}} + N_{\text{PARTIAL}}}{N}$$

$$\text{Faithfulness} = \frac{N_{\text{FULL}}}{N}$$

$$\text{Fabrication Rate} = \frac{N_{\text{FABRICATED}}}{N}$$

$$\text{Hallucination Rate} = \frac{N_{\text{CONTRADICTED}} + N_{\text{UNSUPPORTED}}}{N}$$

### Completeness Identity
$$\text{Faithfulness} + \frac{N_{\text{PARTIAL}}}{N} + \text{Fabrication Rate} + \text{Hallucination Rate} = 1.0$$

> [!NOTE]
> **Source Paper Finding & Metric Sensitivity:**  
> In the reference study (ACI IoT 2023 baseline), standard **Groundedness saturated at $1.0$** across many systems due to lenient context matching. Therefore, **Faithfulness**, **Fabrication Rate**, and **Hallucination Rate** carry the primary statistical discrimination between Traditional ML and DL ensemble detectors.

---

## 5. Worked Examples Table (15 Calibrated Generic-IoT Examples)

| ID | Label | Alert Evidence Snippet ($E$) | Extracted Claim ($c_i$) | Detailed Justification |
| :---: | :--- | :--- | :--- | :--- |
| **EX-01** | `FULL` | `predicted_label: "DDoS_SYN_Flood"`, `confidence: 0.942` | "The detector identified the traffic as a DDoS SYN Flood with 94.2% confidence." | Exact numerical and semantic match with alert prediction fields. |
| **EX-02** | `FULL` | `top_features: [{"rank": 1, "feature_name": "syn_flag_count", "feature_value": 18450.0}]` | "Analysis of the flow reveals an elevated SYN flag count totaling 18,450." | Numerical value and feature name are explicitly identical to rank 1 SHAP entry. |
| **EX-03** | `FULL` | KB Chunk: "TCP SYN floods exhaust connection state tables by sending repeated SYN packets without completing the 3-way handshake." | "The attack mechanism targets connection state table exhaustion via incomplete handshakes." | Directly corroborated by the retrieved knowledge base chunk. |
| **EX-04** | `PARTIAL` | `top_features: [{"feature_name": "dst_to_src_byte_ratio", "feature_value": 0.012}]` | "The inbound and outbound byte ratios are highly asymmetrical." | Conceptually accurate regarding 0.012 ratio, but omits the exact feature name and numerical metric. |
| **EX-05** | `PARTIAL` | `base_learner_votes: [1D_CNN: DDoS_SYN, LSTM: DDoS_SYN, Transformer: Port_Scan]` | "The deep learning models displayed broad agreement regarding the malicious nature of the traffic." | Accurate in spirit (all voted malicious), but glosses over the split vote between DDoS and Port Scan. |
| **EX-06** | `PARTIAL` | `packet_rate_per_sec: 3120.4` | "Network throughput exhibited abnormally rapid packet transmission." | Captures the high rate phenomenon but expresses it as a qualitative generalization. |
| **EX-07** | `UNSUPPORTED`| `predicted_label: "DDoS_SYN_Flood"`, `dataset: "ACI-IoT-2023"` | "The infected IoT device likely belongs to the Mirai botnet family." | No botnet family attribution exists in $E$. The deduction is plausible but uncorroborated; invents no checkable IDs. |
| **EX-08** | `UNSUPPORTED`| `flow_duration_ms: 5912.0` | "The adversary terminated the connection after verifying target responsiveness." | Attributing intent to the adversary is speculative and unsupported by flow duration alone. |
| **EX-09** | `UNSUPPORTED`| Top SHAP feature: `syn_flag_count` (positive attribution) | "The high SYN packet volume was generated by a compromised smart camera gateway." | Hardware device role ('smart camera gateway') is not provided in $E$. |
| **EX-10** | `CONTRADICTED`| `confidence: 0.942`, `detector_correct: true` | "The automated detection engine exhibited low confidence in classifying this incident." | Directly contradicts the 0.942 confidence score in $E_{\text{alert}}$. |
| **EX-11** | `CONTRADICTED`| `top_features: [{"feature_name": "unique_dst_ports_contacted", "feature_value": 1.0}]` | "The flow demonstrates horizontal scanning across hundreds of remote destination ports." | Directly contradicts `unique_dst_ports_contacted == 1.0`. |
| **EX-12** | `CONTRADICTED`| `predicted_label: "DDoS_SYN_Flood"` | "Traffic patterns indicate a benign network surge during scheduled backup operations." | Directly negates the malicious attack classification. |
| **EX-13** | `FABRICATED` | Top-10 features contain: `syn_flag_count`, `packet_rate_per_sec` (no payload features) | "Payload inspection identified an exploit payload matching CVE-2023-1389." | Invents an explicit CVE identifier absent from $E$ (Rule 1). |
| **EX-14** | `FABRICATED` | $E$ contains internal IoT IP `192.168.1.105` | "The command and control server was traced to external IP 198.51.100.42." | External IP address is completely invented and absent from $E$. |
| **EX-15** | `FABRICATED` | Top-10 SHAP features do not include `dns_query_length` | "The primary feature driving detection was dns_query_length with a value of 240." | Invents a feature name not present in the top-10 SHAP attributions in $E$. |

---

## 6. Boundary Cases & Official Rulings

### Boundary Case 1: Plausible but Unlisted Networking Metric
- **Scenario:** The report states: *"The flow exhibited a TCP retransmission rate of 84.2%, causing network degradation."*
- **Evidence ($E$):** $E$ lists `syn_flag_count`, `packet_rate_per_sec`, and `flow_duration_ms`, but **no** feature named `tcp_retransmission_rate`.
- **Ruling:** **`FABRICATED`**.
- **Justification:** Under **Rule 1**, quantitative metrics or feature names not explicitly present in $E$ are invented checkable facts, regardless of whether high retransmission is physically plausible during SYN floods.

### Boundary Case 2: Reconnaissance Attack Labeled as DoS
- **Scenario:** The report states: *"This incident represents an aggressive volumetric denial of service intended to take the endpoint offline."*
- **Evidence ($E$):** `predicted_label: "Port_Scan"`, `top_features: [unique_dst_ports_contacted: 1024, packet_rate: 15.2]`.
- **Ruling:** **`CONTRADICTED`**.
- **Justification:** Port scanning is reconnaissance, not a volumetric denial-of-service attack. The claim directly contradicts the operational characterization of the predicted attack class in $E$.

### Boundary Case 3: Generic Botnet Architectural Trivia
- **Scenario:** The report states: *"Botnets frequently utilize IRC or peer-to-peer protocols for decentralized command dissemination."*
- **Evidence ($E$):** $E$ contains flow metrics for `DDoS_SYN_Flood` with no mention of botnet C2 mechanisms.
- **Ruling:** **`UNSUPPORTED`**.
- **Justification:** The claim contains domain knowledge about botnet architectures generally, but has no grounding in $E$. Because it does not invent an incident-specific entity (such as a specific C2 IP or CVE), it is classified as `UNSUPPORTED` rather than `FABRICATED`.

---

## 7. Judge LLM Prompt Specification

When executing automated evaluation via an LLM judge (default model: `meta-llama/Llama-3.1-8b-instant`), the evaluator must adhere strictly to the system instruction and JSON schema below.

### System Prompt Template
```text
You are an expert cybersecurity evaluation auditor grading LLM incident reports.
You will be provided with:
1. REFERENCE EVIDENCE SET (E): The ground truth alert metadata, top-10 SHAP features, and retrieved knowledge chunks.
2. INCIDENT REPORT: The generated text report to evaluate.

Your task:
1. Extract all atomic claims from the incident report.
2. For each atomic claim, evaluate it strictly against E and assign exactly ONE label:
   - "FULL": 100% corroborated by explicit facts in E.
   - "PARTIAL": Supported in spirit/concept, but over-generalized or imprecise.
   - "UNSUPPORTED": Qualitative deduction or trivia unsupported by E, but invents no specific identifiers or metrics.
   - "CONTRADICTED": Directly conflicts with explicit values or facts in E.
   - "FABRICATED": Asserts an invented specific checkable entity (CVE, IP, MITRE ID, metric, or unlisted feature name) absent from E.

Apply the strict tie-break priority:
FABRICATED > CONTRADICTED > UNSUPPORTED > FULL / PARTIAL.

Output MUST strictly adhere to the following JSON schema:
{
  "claims": [
    {
      "claim_id": <int>,
      "text": "<claim text>",
      "label": "FULL" | "PARTIAL" | "UNSUPPORTED" | "CONTRADICTED" | "FABRICATED",
      "justification": "<concise explanation referencing E>"
    }
  ]
}
```
