from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import ollama
except ImportError as exc:
    raise SystemExit(
        "The 'ollama' Python package is required. Install it in your project environment, "
        "then rerun this script."
    ) from exc

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "limitation": {"type": "string"},
        "operations_not_applied": {
            "type": "array",
            "items": {"type": "string"},
        },
        "supporting_quote": {"type": "string"},
    },
    "required": ["limitation", "operations_not_applied", "supporting_quote"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You extract limitations from scientific-paper excerpts.
Use only information stated in the supplied excerpt. Do not use outside knowledge.
Return exactly the requested JSON object.
- limitation: a concise, faithful statement of the limitation in the excerpt. Do not add unsupported causes, consequences, or recommendations.
- operations_not_applied: include only image/model operations explicitly stated as not applied or omitted. Otherwise return an empty array. Do not list techniques merely mentioned as examples or remedies.
- supporting_quote: copy one exact, contiguous quote from the excerpt that supports the limitation. Do not paraphrase or repair the quote.
- If the excerpt does not explicitly state a limitation, set limitation to exactly "NO_EXPLICIT_LIMITATION", operations_not_applied to [], and supporting_quote to an empty string.
"""


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).replace("\u00ad", "")
    # Normalize common PDF line-wrap hyphenation and whitespace for quote checks.
    value = re.sub(r"(?<=\w)-\s+(?=\w)", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip().casefold()


def schema_valid(obj: Any) -> bool:
    return (
        isinstance(obj, dict)
        and set(obj) == {"limitation", "operations_not_applied", "supporting_quote"}
        and isinstance(obj.get("limitation"), str)
        and isinstance(obj.get("operations_not_applied"), list)
        and all(isinstance(x, str) for x in obj["operations_not_applied"])
        and isinstance(obj.get("supporting_quote"), str)
    )


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate structured limitation extraction with local Ollama.")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases (0 = all).")
    parser.add_argument("--timeout", type=float, default=90.0, help="Per-call HTTP timeout in seconds.")
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=2048)
    parser.add_argument("--host", default="http://localhost:11434")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    cases = load_cases(script_dir / "limitation_eval_cases.jsonl")
    if args.limit > 0:
        cases = cases[:args.limit]

    client = ollama.Client(host=args.host, timeout=args.timeout)
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    output_dir = script_dir.parent.parent / "storage" / "evaluation" / "extraction" / "limitation_eval_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"limitation_eval_{args.model.replace(':', '-')}_{stamp}.json"
    csv_path = json_path.with_suffix(".csv")

    for index, case in enumerate(cases, 1):
        prompt = (
            "Extract the limitation from this excerpt.\n\n"
            f"SOURCE: {case['source_paper']}\n"
            f"SECTION: {case['section']}\n"
            f"EXCERPT:\n{case['excerpt']}"
        )
        started = time.perf_counter()
        raw = ""
        parsed: Any = None
        error = ""
        try:
            response = client.chat(
                model=args.model,
                think=False,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                format=SCHEMA,
                options={
                    "temperature": 0,
                    "num_ctx": args.num_ctx,
                    "num_predict": args.num_predict,
                },
            )
            raw = response.message.content or ""
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                error = f"JSON parse error: {exc}"
        except Exception as exc:  # retain failed cases and continue the run
            error = f"{type(exc).__name__}: {exc}"

        elapsed = time.perf_counter() - started
        latencies.append(elapsed)
        valid = schema_valid(parsed)
        quote_ok = False
        operations_ok = False
        if valid:
            quote = parsed["supporting_quote"]
            quote_ok = (
                parsed["limitation"] == "NO_EXPLICIT_LIMITATION" and quote == ""
                if case["gold_limitation"] == "NO_EXPLICIT_LIMITATION"
                else bool(quote) and normalize_text(quote) in normalize_text(case["excerpt"])
            )
            predicted_ops = sorted({x.strip().casefold() for x in parsed["operations_not_applied"]})
            expected_ops = sorted({x.strip().casefold() for x in case["gold_operations_not_applied"]})
            operations_ok = predicted_ops == expected_ops

        row = {
            "id": case["id"],
            "source_paper": case["source_paper"],
            "section": case["section"],
            "page": case.get("page"),
            "excerpt": case["excerpt"],
            "gold_limitation": case["gold_limitation"],
            "gold_operations_not_applied": case["gold_operations_not_applied"],
            "gold_concepts": case["gold_concepts"],
            "prediction": parsed,
            "raw_response": raw,
            "json_valid": parsed is not None,
            "schema_valid": valid,
            "quote_fidelity": quote_ok if valid else False,
            "operations_exact": operations_ok if valid else False,
            "latency_seconds": round(elapsed, 3),
            "manual_limitation_correct": None,
            "manual_unsupported_claims": None,
            "review_notes": "",
            "error": error,
        }
        rows.append(row)
        print(
            f"[{index}/{len(cases)}] {case['id']} "
            f"json={row['json_valid']} schema={valid} quote={row['quote_fidelity']} "
            f"ops={row['operations_exact']} latency={elapsed:.2f}s"
        )

        # Checkpoint after every case so a long run retains completed results.
        valid_count_so_far = sum(bool(r["schema_valid"]) for r in rows)
        quote_count_so_far = sum(bool(r["quote_fidelity"]) for r in rows)
        operations_count_so_far = sum(bool(r["operations_exact"]) for r in rows)
        payload = {
            "model": args.model,
            "host": args.host,
            "settings": {"temperature": 0, "num_ctx": args.num_ctx, "num_predict": args.num_predict, "timeout": args.timeout},
            "case_count": len(rows),
            "summary": {
                "schema_valid_rate": valid_count_so_far / len(rows),
                "quote_fidelity_rate": quote_count_so_far / len(rows),
                "operations_exact_rate": operations_count_so_far / len(rows),
                "mean_latency_seconds": statistics.mean(latencies),
                "median_latency_seconds": statistics.median(latencies),
            },
            "results": rows,
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            fields = [
                "id", "source_paper", "section", "page", "gold_limitation", "prediction",
                "json_valid", "schema_valid", "quote_fidelity", "operations_exact", "latency_seconds",
                "manual_limitation_correct", "manual_unsupported_claims", "review_notes", "error",
            ]
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for item in rows:
                serializable = dict(item)
                serializable["prediction"] = json.dumps(item["prediction"], ensure_ascii=False) if item["prediction"] is not None else ""
                writer.writerow(serializable)

    valid_count = sum(bool(r["schema_valid"]) for r in rows)
    quote_count = sum(bool(r["quote_fidelity"]) for r in rows)
    ops_count = sum(bool(r["operations_exact"]) for r in rows)
    print("\nPILOT SUMMARY")
    print(f"Cases: {len(rows)}")
    print(f"Schema-valid: {valid_count}/{len(rows)} ({valid_count / len(rows):.1%})")
    print(f"Quote fidelity: {quote_count}/{len(rows)} ({quote_count / len(rows):.1%})")
    print(f"Operations exact: {ops_count}/{len(rows)} ({ops_count / len(rows):.1%})")
    print(f"Mean latency: {statistics.mean(latencies):.2f}s")
    print(f"Median latency: {statistics.median(latencies):.2f}s")
    print(f"Results: {json_path}")
    print(f"Review CSV: {csv_path}")
    print("Limitation accuracy remains a human-review field in the CSV; do not treat quote/schema checks as factual accuracy.")


if __name__ == "__main__":
    main()
