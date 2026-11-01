# Atlas Network Evidence Assistant

A RAG demo for exploring a fictional two-site network. Ask about designs,
configuration records and synthetic failure scenarios, inspect retrieved sources,
and optionally generate a cited draft.

## Run locally

Requires Python 3.12+ and uv.

```powershell
uv sync
uv run python check_product.py
uv run streamlit run app.py
```

Open the local URL printed by Streamlit. Start with **Local keyword search**;
this needs no retrieval key. Enter a Nebius API key in the password field only
if you want a generated draft. One click makes one paid generation call.

Hybrid search additionally requires a Pinecone key and the populated index and
namespace configured in `rag_settings.py`. The existing upload scripts are
separate setup tools; the app does not upload or modify remote vectors.
A cloned repository does not automatically have access to the original index.

## Demo

1. Choose the FAIL-08 management-services example and set the evidence limit
   to 20. Click **Find evidence**, then inspect the four log sections in Sources.
2. Enter a Nebius key and click **Generate cited draft**. Compare claims with
   cited sources. A narrow local host-role check may withhold an erroneous draft.
3. Choose the FAIL-99 example. Click **Find evidence**. The app reports missing
   evidence and disables generation.

The offline check covers those two retrieval/routing paths. Browser behavior,
hybrid connectivity and generated answers must be checked separately by the user.
No passing test results are claimed until the commands are run.

## How it works

`corpus/` → document loading → structure-aware chunks in `work/chunks.jsonl`
→ local BM25 or Pinecone + BM25 with reciprocal-rank fusion → evidence package
→ LangGraph coverage routing → optional Nebius draft → local checks → source display.

The app reuses the existing pipeline modules through `product_pipeline.py`.
Evidence retains chunk IDs, source paths, exact line ranges, document type and
synthetic-log context. Package-relative citation labels map answers to sources.
Evaluation fixtures remain outside the ingested corpus.

## Limits and data handling

- All network content is fictional training material. Synthetic event times are
  assigned examples, not hardware measurements or SLA evidence.
- Answers are unverified drafts. Valid labels and retrieved coverage do not prove
  factual support. The experimental model reviewer produced a false pass and is
  not used as a product correctness gate.
- Explicit FAIL-xx requests require all available scenario log chunks. This is a
  conservative coverage rule, not semantic sufficiency checking. Broad questions
  without a scenario ID do not get that scenario coverage check.
- The host-role rule detects a narrow known NTP wording pattern; it is not general
  semantic validation. Inspect the cited evidence before relying on a draft.
- Keys stay in app session memory; the app does not write them to disk or include
  them in downloads. Use trusted local hosting and avoid recording keys in videos.
- Hybrid search sends a query to Pinecone. Generation sends the question and
  selected evidence to Nebius. Provider usage and data policies apply.
- Retrieval can miss relevant chunks. Increase evidence selection or refine a
  question when coverage is incomplete. The app does not automatically retry.

## Main files

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `product_pipeline.py` | Retrieval, routing and one-call draft orchestration |
| `prepare_evidence.py` | Full evidence and citation provenance |
| `retrieve_bm25.py`, `retrieve_hybrid.py` | Keyword and hybrid retrieval |
| `route_evidence.py` | LangGraph coverage routes |
| `check_host_roles.py` | Narrow topology-based NTP check |
| `check_product.py` | Two offline product checks |
| `evaluation/` | Excluded evaluation material and known failures |

Build explanation and video outline: `PRODUCT-BUILD.txt`.
