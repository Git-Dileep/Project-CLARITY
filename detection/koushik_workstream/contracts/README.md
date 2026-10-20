# Alert Schema Contract Reference

The authoritative shared alert schema for Project CLARITY is located at the root of the repository:
`contracts/alert_schema_v1.json`

## Rules
1. **Never modify, weaken, or replace** `contracts/alert_schema_v1.json`.
2. All alert exports from `detection/koushik_workstream` must strictly conform to `AlertSchemaV1` Draft 2020-12.
3. Every exported alert payload must satisfy:
   - `identity.alert_id`: format `^ALT-[0-9]{8}-[0-9]{4}$`
   - `identity.detector_family`: `"traditional_ml"` or `"dl_ensemble"`
   - `identity.split`: `"pilot"` or `"test"`
   - `identity.timestamp`: valid ISO 8601 date-time string
   - `prediction.base_learner_votes`: array of **exactly 3** base learners
   - `shap_attribution.explainer_type`: `"TreeExplainer"`, `"DeepExplainer"`, or `"GradientExplainer"`
   - `shap_attribution.top_features`: array of **exactly 10** features ranked 1 through 10
   - `evaluation_metadata`: contains `true_label`, `in_train_leakage_flag`, and `detector_correct` (excluded from LLM generation prompts)
