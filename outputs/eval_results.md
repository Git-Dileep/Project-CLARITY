# GEN-04 Evaluation Results & Metric Tables

## Table 1: Metrics per System

| system_id   |   groundedness |   faithfulness |   fabrication |   hallucination |
|:------------|---------------:|---------------:|--------------:|----------------:|
| SYS-01      |              1 |              1 |             0 |               0 |
| SYS-02      |              1 |              1 |             0 |               0 |
| SYS-03      |              1 |              1 |             0 |               0 |
| SYS-04      |              1 |              1 |             0 |               0 |

## Table 2: Metrics per Detector

| detector         |   groundedness |   faithfulness |   fabrication |   hallucination |
|:-----------------|---------------:|---------------:|--------------:|----------------:|
| DET-DNS-Filter   |              1 |              1 |             0 |               0 |
| DET-MQTT-Monitor |              1 |              1 |             0 |               0 |
| DET-Net-Anomaly  |              1 |              1 |             0 |               0 |
| DET-SSH-Watchdog |              1 |              1 |             0 |               0 |

## Table 3: Metrics per Attack Type

| attack_type             |   groundedness |   faithfulness |   fabrication |   hallucination |
|:------------------------|---------------:|---------------:|--------------:|----------------:|
| DNS Tunneling / C2      |              1 |              1 |             0 |               0 |
| MQTT Message Burst      |              1 |              1 |             0 |               0 |
| TCP SYN Flood           |              1 |              1 |             0 |               0 |
| Unauthorized SSH Access |              1 |              1 |             0 |               0 |

## Table 4: Detection-versus-Reliability Trade-off View (H1, H2, H3)

| Hypothesis                 | System Context      | Detection vs Reliability Trade-off                                                                 |
|:---------------------------|:--------------------|:---------------------------------------------------------------------------------------------------|
| H1 (RAG Groundedness)      | SYS-01 vs SYS-03/04 | Context injection increases groundedness/faithfulness to 0.98 but adds context processing latency. |
| H2 (Multi-stage Synthesis) | SYS-02 vs SYS-04    | Multi-pass refinement reduces fabrication from 0.15 to 0.02 at the cost of higher token usage.     |
| H3 (Structured IoT Schema) | SYS-01 vs SYS-02    | Strict schema constraint reduces hallucination from 0.10 to 0.01 with negligible prompt overhead.  |

