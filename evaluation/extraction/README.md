# Limitation extraction evaluation pilot

This is a model-level extraction pilot for the existing three-field output:
`limitation`, `operations_not_applied`, and `supporting_quote`.

## Cases

- 20 excerpt-level cases from the brain-tumor papers already in the project corpus.
- Cases cover omitted augmentation, data scarcity, heterogeneity, imbalance, annotation and resolution inconsistencies, preprocessing variation, missing treatment information, single-timepoint data, prospective validation, interpretability, overfitting, compute requirements, domain shift, feature-selection bias, image appearance, single-dataset generalization, and a negative control.
- The excerpts are supplied to the model individually. The script does not run PDF parsing or the full ingestion pipeline.

## Run

Place both files in `M:\Projects\ResearchPaper-AgenticRAG\evaluation\extraction\` (create the folder if needed), then run from the project root:

```powershell
uv run python evaluation/extraction/run_limitation_eval.py --limit 1
```

If the one-case smoke test completes, run all 20:

```powershell
uv run python evaluation/extraction/run_limitation_eval.py
```

Optional settings:

```powershell
uv run python evaluation/extraction/run_limitation_eval.py --model qwen3:4b --timeout 90 --num-ctx 4096 --num-predict 2048
```

The script calls the local Ollama API directly, with temperature 0 and the same JSON fields. It records one checkpoint after every case under `storage/evaluation/extraction/limitation_eval_runs/`, producing JSON and CSV results. No external API or model download is used.

## What is measured

- **JSON/schema validity:** automatically checked.
- **Quote fidelity:** automatically checks that the returned quote is present in the supplied excerpt after whitespace/PDF-hyphen normalization. The negative control expects an empty quote.
- **Operations:** exact set comparison against the gold list. Only the explicit augmentation-omission case expects `rotation` and `cropping`; all others expect an empty list.
- **Latency:** end-to-end wall time per call, plus mean and median.
- **Limitation factual accuracy:** requires human review. In the CSV, fill `manual_limitation_correct` with `yes`/`no` and `manual_unsupported_claims` with `yes`/`no`; put rationale in `review_notes`. A concise paraphrase is acceptable if it preserves the gold concepts and does not add unsupported claims.

This is a small pilot set, not a final publication-grade benchmark. It is intentionally excerpt-level; it does not measure extraction from complete PDFs or omitted-context errors.
