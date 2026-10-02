# Comprehensive Repository Audit — Universal Agentic Research Paper Assistant (`BrainTumourAgenticRAG`)

**Audit Pass**: Complete Two-Pass Comprehensive Audit
**Date**: October 2026
**Auditor**: Senior Software Architect, Research Engineer & Independent AI Systems Reviewer

---

## 1. Executive Summary & Audit Overview

This report completes a comprehensive, evidence-based audit of the `BrainTumourAgenticRAG` repository. The evaluation covers project objectives, architecture, data flows, document parsing, retrieval and reranking pipelines, evaluation methodology, agent tool integration, and readiness for future multi-step research workflows.

### Summary of System Status:
- **Core Retrieval & Reranking Architecture**: The system possesses a sound hybrid retrieval (Dense ChromaDB + BM25 with compound-token aware tokenization, fused via Reciprocal Rank Fusion) and cross-encoder reranking foundation (`BAAI/bge-reranker-base`, `MiniLM-MSMARCO`).
- **Agent Orchestration & Tooling**: The system is currently a single-tool retriever rather than a Universal Agentic Assistant. Only one tool (`retrieve_papers`) is registered in `agent/tool_registry.py`, driven by a basic linear ReAct orchestrator loop (`agent/orchestrator.py`). Multi-step research workflows, paper comparison tools, table/figure retrieval tools, and data visualization execution environments are non-existent in code.
- **Ingestion & Extraction Engine**: The PDF ingestion pipeline contains significant engineering complexity (~3,500 LOC across GROBID XML parsing, Docling artifact extraction, PyMuPDF layout parsing, and a 1,200 LOC Tesseract/rule-based OCR table fallback engine). However, this operates on an active corpus of only 6 papers, creating high maintenance overhead relative to corpus size.
- **Evaluation & Benchmark Integrity**: Benchmark evaluation (`retrieval_benchmark_v1.json`) relies on sparse qrel judgments (73 relevance annotations across 40 queries; 36 queries are paper-scoped). Furthermore, raw execution output files for dense/hybrid and reranking comparisons are absent from the storage path, meaning benchmark score claims cannot be independently verified from stored run artifacts.
- **Code Health & Reliability**: The codebase contains duplicate LLM client abstractions (`llm/base.py` vs `agent/models/base.py`) and a critical syntax defect caused by a Byte Order Mark (`\xef\xbb\xbf`) on line 1 of `extraction/pipeline.py`.

---

## 2. Capabilities Matrix: Implemented vs. Planned

| Capability | Status | Implementation Analysis & Gaps |
| :--- | :--- | :--- |
| **Grounded Question Answering** | **Partial** | Single-tool passage retrieval available. Prompt relies on system message instructions; no deterministic citation verifier or answer grounder. |
| **Semantic & Lexical Retrieval** | **Verified Implemented** | ChromaDB vector search + BM25 lexical search with Reciprocal Rank Fusion ($k=60$). |
| **Cross-Encoder Reranking** | **Verified Implemented** | `CrossEncoderReranker` with token windowing (256/384/512 max length) and max window aggregation. |
| **Paper-Scoped Retrieval** | **Verified Implemented** | Scope filtering by `paper_id` in Chroma metadata and BM25 candidate selection. |
| **Structured PDF Ingestion** | **Verified Implemented** | Ingestion pipeline combining GROBID TEI XML, Docling, PyMuPDF, and OCR fallback. |
| **Structured Table & Figure Tool** | **Unimplemented** | Captions are chunked as text, but structured table matrices (`tables` key in canonical JSON) and figures are not exposed to agent tools. |
| **Paper Comparison & Synthesis** | **Unimplemented** | No multi-paper matrix comparison tools or comparative extraction workflows exist in the agent system. |
| **Quantitative Analysis & Visualization** | **Unimplemented** | No sandboxed Python execution tool, pandas data frames, or chart generation tools. |
| **Workflow-First Orchestrator** | **Minimal** | Basic linear ReAct loop (`agent/orchestrator.py`). Lacks DAG execution, state management, or retry routing. |
| **Formula & Equation Analysis** | **Planned (Later)** | Formula regions protected during cleaning (`normalizer.py`); no specialized formula index or solver tool. |

---

## 3. Data Flow & Architectural Analysis

### Data Flow Diagram

```
[Raw PDF Papers] ──> [Ingestion Pipeline: GROBID + Docling + PyMuPDF + OCR Fallback]
                                       │
                                       ▼
                       [Canonical JSON Store (paper_XXX.json)]
                                       │
                        ┌──────────────┴──────────────┐
                        ▼                             ▼
              [Text Normalizer]            [Structured Tables & Figures]
                        │                    (Stored, but unindexed/unused)
                        ▼
               [Text Chunking]
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
 [ChromaDB Vector Store]       [In-Memory BM25 Index]
         │                             │
         └──────────────┬──────────────┘
                        ▼
            [Research Retriever (RRF)]
                        │
                        ▼
            [Cross-Encoder Reranker]
                        │
                        ▼
         [Agent Tool: retrieve_papers] ──> [ReAct Orchestrator] ──> [LLM Answer]
```

---

## 4. Detailed Audit Findings by Category

### A. Ingestion, Parsing, and Canonical Document Representation
1. **Critical Defect — Ingestion Pipeline BOM Bug (`CRIT-01`)**:
   - *Location*: `extraction/pipeline.py:1`
   - *Details*: File begins with a UTF-8 Byte Order Mark (`\xef\xbb\xbf# extraction/pipeline.py`), causing standard Python imports and syntax tools to fail.
2. **High Severity — Disconnect Between Canonical Data and Retrieval (`HIGH-02`)**:
   - *Location*: `chunking/text_chunker.py`, `retrieval/chroma_retriever.py`
   - *Details*: Canonical JSON files preserve rich structured tables (`tables` key) and figure metadata. However, `text_chunker.py` extracts only `table_caption` and `figure_caption` as text strings. The actual row/column cells and values are discarded during chunking, preventing any table QA.
3. **Medium Severity — Extreme Ingestion Pipeline Overhead (`MED-01`)**:
   - *Location*: `extraction/grobid_parser.py` (1,242 LOC), `extraction/table_fallback.py` (1,201 LOC), `extraction/docling_artifacts.py` (787 LOC)
   - *Details*: ~3,500 LOC are devoted to multi-parser orchestration and OCR table reconstruction for a 6-paper corpus. Hardcoded external dependencies (e.g. `http://localhost:8070` for GROBID, local Tesseract installation) increase system brittleness without adding proportional quality for standard PDFs.

### B. Retrieval & Reranking Review
1. **High Severity — Sparse Benchmark Labels & Artificial Scoping (`HIGH-01`)**:
   - *Location*: `storage/evaluation/retrieval_metrics/retrieval_benchmark_v1.json`
   - *Details*: The benchmark contains 40 queries, but only 73 total relevant chunk annotations (1.8 chunks per query). 36 out of 40 queries are scoped to a single paper (`scope_paper_id`).
   - *Impact*: Candidate pools are artificially restricted to ~50 chunks, obscuring true corpus-wide dense vs. lexical retrieval behavior and inflating metrics. Unlabeled relevant chunks in the corpus act as false negatives during recall evaluation.
2. **Observation — Reranker Long-Passage Windowing Strength (`OBS-01`)**:
   - *Location*: `reranking/cross_encoder.py`
   - *Details*: `CrossEncoderReranker` correctly handles passages exceeding tokenizer context limits by slicing into overlapping windows and taking the maximum score across windows.

### C. Agent Tools & Orchestration
1. **Critical Defect — Complete Absence of Planned Research Tools (`CRIT-02`)**:
   - *Location*: `agent/tool_registry.py`, `agent/orchestrator.py`
   - *Details*: The tool registry contains exactly one tool (`retrieve_papers`). Required research capabilities—such as metadata lookup, structured table extraction, paper comparison matrix generation, data analysis, and chart visualization—are completely absent.
2. **High Severity — Duplicate LLM Abstraction Layers (`HIGH-03`)**:
   - *Location*: `llm/base.py` vs `agent/models/base.py`
   - *Details*: The repository maintains two parallel, non-overlapping LLM client abstractions: `llm/base.py` (`ChatModel`) for extraction scripts, and `agent/models/base.py` (`ChatModel`) for the agent orchestrator.

---

## 5. Proposed Target Architecture

To realize the Universal Agentic Research Paper Assistant vision, the architecture should evolve into a **Workflow-First Research System**:

```
+-----------------------------------------------------------------------------------+
|                                 User Task / Input                                 |
+-----------------------------------------------------------------------------------+
                                          │
                                          ▼
+-----------------------------------------------------------------------------------+
|                       Stateful Research Orchestrator                              |
|         (Deterministic Stage Router & Structured Execution Plan)                  |
+-----------------------------------------------------------------------------------+
        │                      │                      │                      │
        ▼                      ▼                      ▼                      ▼
+───────────────+      +───────────────+      +───────────────+      +───────────────+
| Passage       |      | Structured    |      | Paper         |      | Python Data   |
| Retriever     |      | Table & Figure|      | Comparison    |      | Analysis &    |
| Tool          |      | Reader Tool   |      | Matrix Tool   |      | Plotting Tool |
+───────────────+      +───────────────+      +───────────────+      +───────────────+
        │                      │                      │                      │
        └──────────────────────┼──────────────────────┴──────────────────────┘
                               │
                               ▼
+-----------------------------------------------------------------------------------+
|                        Unified Storage & Registry                                 |
|     (Canonical JSON + Chunks + Structured Table Matrices + Metadata DB)           |
+-----------------------------------------------------------------------------------+
```

### Architectural Key Principles:
1. **Unified LLM Module**: Merge `llm/` and `agent/models/` into a single, clean LLM client layer.
2. **Structured Table Chunking**: Index tabular data as Markdown/CSV blocks in ChromaDB alongside text chunks.
3. **Deterministic Python Execution Tool**: Provide a sandboxed Python execution tool (using pandas & matplotlib) for calculations, statistics, and chart generation.
4. **Stateful Research Agent Workflow**: Replace the loop in `orchestrator.py` with explicit research state tracking (e.g. Query $\rightarrow$ Retrieval $\rightarrow$ Extraction $\rightarrow$ Matrix Comparison $\rightarrow$ Synthesis).

---

## 6. Assessment of Evidence & Artifacts

- **Stored Run Results**: Evaluation output JSON files for historical dense vs. hybrid comparisons and reranking runs were not present in `storage/evaluation/retrieval_metrics/`.
- **Limitation Extraction Evaluation**: As confirmed during audit interactions, the previous LLM limitation extraction attempt (`storage/evaluation/extraction/limitation_eval_runs/`) was flawed and abandoned.
- **Verification Status**: Code structure and algorithms (BM25 tokenization, RRF fusion, windowed reranking) are verified via static code inspection and test suites (`tests/test_hybrid_retrieval.py`, `tests/test_reranking.py`). Benchmark result claims remain unverified due to missing run logs.

---

## 7. Prioritized Remediation Roadmap

```
Phase 1: Stabilization & Refactoring (Immediate)
├── Fix extraction/pipeline.py BOM bug
├── Consolidate LLM abstractions into single module
└── Simplify PDF ingestion dependencies

Phase 2: Data Representation & Tool Expansion (Short-Term)
├── Enable Structured Table Chunking & Tool Access
├── Register Paper Metadata & Comparison Tools
└── Implement Sandboxed Python Analysis & Plotting Tool

Phase 3: Workflow Orchestration & Grounded Synthesis (Medium-Term)
├── Upgrade Agent Orchestrator to Stateful Workflow Engine
└── Implement Citation Verification & Grounding Guardrails

Phase 4: Benchmark Hardening (Long-Term)
└── Expand Benchmark qrels & Multi-Paper QA Evaluation Suite
```

---

## 8. Next Development Actions & Acceptance Criteria

### Action 1: Fix Pipeline Syntax Defect & Consolidate LLM Abstraction Layer
- **Task**: Remove the Byte Order Mark from `extraction/pipeline.py`. Merge `llm/base.py` and `agent/models/base.py` into a unified `llm/` abstraction layer supporting structured outputs and tool calling.
- **Acceptance Criteria**:
  1. `extraction/pipeline.py` parses clean without BOM encoding errors.
  2. All agent and extraction code imports from a single `llm/` module.
  3. `pytest` runs without module import errors.

### Action 2: Expose Structured Table Matrices to Agent Tools
- **Task**: Create a `TableChunker` that converts canonical JSON table structures into searchable Markdown/CSV text chunks, and implement a `get_paper_table` agent tool.
- **Acceptance Criteria**:
  1. Table rows/columns are searchable via `ResearchRetriever`.
  2. A new tool `get_paper_table(paper_id, table_id)` returns the exact structured table matrix.
  3. Unit test verifies structured table QA lookup.

### Action 3: Implement Deterministic Python Data Analysis & Plotting Tool
- **Task**: Build a `PythonAnalysisTool` that accepts Python code snippets, executes them in an isolated environment with pandas/matplotlib, and returns output data and saved image paths.
- **Acceptance Criteria**:
  1. Tool accepts tabular data/lists and executes summary calculations deterministically.
  2. Generates and saves chart artifacts (e.g. `.png`) to `storage/artifacts/`.
  3. Unit test verifies data aggregation and chart creation.

### Action 4: Build Paper Comparison Tool & Upgrade Orchestrator Workflow
- **Task**: Create a `ComparePapersTool` for extracting key methodologies, datasets, and results across multiple papers into a comparative schema.
- **Acceptance Criteria**:
  1. Tool accepts a list of `paper_ids` and comparison dimensions.
  2. Returns a structured JSON comparison matrix.
  3. Agent orchestrator can execute a 2-step research workflow (Retrieve $\rightarrow$ Compare).
