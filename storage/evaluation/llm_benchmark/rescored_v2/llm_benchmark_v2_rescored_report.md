# LLM Benchmark V2 — Corrected Offline Rescore

No model inference was run for this report; saved V2 outputs were rescored offline.

## Summary

Query analysis retains the V2 gold-label scoring. Extraction now reports record precision/recall/F1 separately from entity, field, value, unit, condition and quote dimensions. Condition wording no longer determines whether the core fact record is correct.

Synthesis reports required-answer-key coverage, required evidence coverage, support-ID validity and numeric fidelity separately. `strict_pass_rate` is a heuristic task pass, not a proof of semantic entailment.

| Model | Query score | Fact core micro-F1 | Fact verified micro-F1 | Fact macro-F1 | Quote fidelity | Synthesis key-answer rate | Synthesis strict pass | Median latency (analysis / extraction / synthesis) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3:4b-instruct | 0.943 | 0.880 | 0.800 | 0.800 | 1.000 | 0.800 | 0.800 | 5.55s / 7.07s / 4.48s |
| granite4:3b | 0.857 | 0.182 | 0.091 | 0.160 | 0.500 | 0.200 | 0.200 | 3.84s / 6.60s / 3.65s |
| phi4-mini | 0.943 | 0.476 | 0.381 | 0.433 | 0.800 | 0.600 | 0.600 | 4.49s / 4.73s / 4.44s |
| qwen3.5:4b | 0.971 | 0.240 | 0.240 | 0.227 | 1.000 | 0.600 | 0.600 | 5.41s / 8.90s / 5.51s |
| qwen3.5:9b | 0.886 | 0.522 | 0.522 | 0.533 | 0.800 | 0.800 | 0.800 | 14.55s / 38.36s / 21.56s |
| gemma3:4b | 0.800 | 0.560 | 0.320 | 0.500 | 0.700 | 0.600 | 0.600 | 4.89s / 8.95s / 5.38s |
| granite4:7b-a1b-h | — | — | — | — | — | — | — | unavailable: show_failed: {"error":"model 'granite4:7b-a1b-h' not found"} |

## Interpretation safeguards

- Only five cases per task were used; treat all rankings as pilot evidence, not a statistically reliable final model selection.
- Core record F1 requires a matched entity, field, value and semantically represented unit. Verified record F1 additionally requires a source-grounded quote containing the extracted value. Free-form condition mismatch is reported separately and does not erase a correct core fact.
- Quote fidelity is normalized substring containment (case/punctuation/whitespace tolerant), not byte-for-byte equality.
- Support-ID validity only checks that each support ID exists in the provided evidence. It cannot establish semantic entailment; manually review synthesis claims, especially extra claims.
- Query-analysis gold annotations are unchanged and may need a human review before the next formal benchmark.

## Detailed output files

- Per-case details: `llm_benchmark_v2_rescored_cases.json`
- Machine-readable summary: `llm_benchmark_v2_rescored_summary.json`
