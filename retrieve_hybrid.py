"""Lesson 11: combine dense and BM25 rankings with reciprocal rank fusion.

Preview: uv run python retrieve_hybrid.py
Search: uv run python retrieve_hybrid.py --execute

Execution builds BM25 locally, embeds one question through LangChain, searches
Pinecone read-only, and fuses candidates by chunk ID. No answer is generated,
records written, or files changed. Embedding/search usage may be billed.
"""

import argparse
import getpass
import math
import re
import warnings
from pathlib import Path

from check_chunk_lengths import DEFAULT_EXPORT, embedding_text, read_chunks
from embed_one_chunk import validate_vector
from rag_settings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, INDEX_NAME, NAMESPACE
from retrieve_bm25 import bm25_results, build_bm25, network_tokens
from retrieve_dense import DEFAULT_QUESTION
from setup_pinecone_index import check_configuration
from upload_one_chunk import storage_metadata


RRF_CONSTANT = 60


def explicit_scenarios(question: str) -> list[str]:
    """Recognise literal scenario IDs only; do not infer IDs from descriptions."""
    return sorted(set(re.findall(r"\bFAIL-\d{2}\b", question.upper())))


def scope_records(records: list[dict], scenario_ids: list[str]) -> list[dict]:
    """Keep all non-log evidence and only the explicitly requested log scenarios."""
    if not scenario_ids:
        return records
    return [
        record for record in records
        if record["metadata"]["document_type"] != "synthetic_log"
        or record["metadata"].get("scenario_id") in scenario_ids
    ]


def dense_filter(scenario_ids: list[str]) -> dict:
    """Apply the same evidence scope remotely, before Pinecone ranks results."""
    conditions = [
        {"embedding_model": {"$eq": EMBEDDING_MODEL}},
        {"embedding_dimension": {"$eq": EMBEDDING_DIMENSIONS}},
        {"embedding_input_format": {"$eq": "source-labels-context-evidence-v1"}},
    ]
    if scenario_ids:
        conditions.append({"$or": [
            {"document_type": {"$ne": "synthetic_log"}},
            {"$and": [
                {"document_type": {"$eq": "synthetic_log"}},
                {"scenario_id": {"$in": scenario_ids}},
            ]},
        ]})
    return {"$and": conditions}


def fuse_rankings(dense: list[dict], lexical: list[dict]) -> list[dict]:
    """Equal-weight RRF: each method contributes 1/(60 + one-based rank)."""
    merged = {}
    for method, candidates in (("dense", dense), ("bm25", lexical)):
        seen = set()
        for rank, candidate in enumerate(candidates, start=1):
            chunk_id = candidate["id"]
            if chunk_id in seen:
                raise ValueError(f"Duplicate ID in {method} candidate list.")
            seen.add(chunk_id)
            row = merged.setdefault(chunk_id, {
                "id": chunk_id, "record": candidate["record"], "rrf_score": 0.0,
                "dense_rank": None, "bm25_rank": None,
                "dense_score": None, "bm25_score": None,
                "dense_contribution": 0.0, "bm25_contribution": 0.0,
            })
            if row["record"] != candidate["record"]:
                raise ValueError("The two retrievers disagree about the same chunk's evidence.")
            contribution = 1.0 / (RRF_CONSTANT + rank)
            row[f"{method}_rank"] = rank
            row[f"{method}_score"] = candidate["score"]
            row[f"{method}_contribution"] = contribution
            row["rrf_score"] += contribution

    # IDs provide a deterministic tiebreaker; raw scores never enter fusion.
    return sorted(merged.values(), key=lambda row: (-row["rrf_score"], row["id"]))


def dense_candidates(
    records: list[dict], question: str, api_key: str, count: int,
    scenario_ids: list[str] | None = None,
) -> list[dict]:
    """Get Pinecone candidates and verify them against the local export."""
    from langchain_pinecone import PineconeEmbeddings
    from pinecone import Pinecone

    by_id = {record["id"]: record for record in records}
    client = Pinecone(api_key=api_key)
    description = client.describe_index(INDEX_NAME).to_dict()
    check_configuration(description)
    if description.get("status", {}).get("ready") is not True:
        raise ValueError("Index is not ready.")
    embeddings = PineconeEmbeddings(
        model=EMBEDDING_MODEL,
        dimension=EMBEDDING_DIMENSIONS,
        pinecone_api_key=api_key,
        document_params={"input_type": "passage", "dimension": EMBEDDING_DIMENSIONS, "truncate": "NONE"},
        query_params={"input_type": "query", "dimension": EMBEDDING_DIMENSIONS, "truncate": "NONE"},
    )
    vector = validate_vector([embeddings.embed_query(question)])
    response = client.Index(host=description["host"]).query(
        vector=vector, top_k=count, namespace=NAMESPACE,
        include_metadata=True, include_values=False,
        filter=dense_filter(scenario_ids or []),
    )
    candidates = []
    seen = set()
    for match in response.matches or []:
        if match.id not in by_id or match.id in seen:
            raise ValueError("Dense search returned an unknown or duplicate chunk ID.")
        record = by_id[match.id]
        if match.metadata != storage_metadata(record, embedding_text(record)):
            raise ValueError("Remote evidence differs from the local export.")
        if match.score is None or not math.isfinite(match.score):
            raise ValueError("Invalid dense similarity score.")
        seen.add(match.id)
        candidates.append({"id": match.id, "score": float(match.score), "record": record})
    return candidates


def show_results(rows: list[dict], top_k: int, full_text: bool) -> None:
    print(f"\nTop {min(top_k, len(rows))} fused candidates:")
    for position, row in enumerate(rows[:top_k], start=1):
        record = row["record"]
        metadata = record["metadata"]
        ranges = metadata.get("source_line_ranges") or [f"{metadata['line_start']}-{metadata['line_end']}"]
        print(f"\n{position}. RRF score: {row['rrf_score']:.6f}")
        print(f"   Dense rank: {row['dense_rank']} | BM25 rank: {row['bm25_rank']} (None = absent)")
        print(f"   Contributions: dense={row['dense_contribution']:.6f}, BM25={row['bm25_contribution']:.6f}")
        print(f"   Raw scores (not added): dense={row['dense_score']}, BM25={row['bm25_score']}")
        print(f"   Source: {metadata['source']} | exact lines: {', '.join(ranges)}")
        print(f"   Type: {metadata['document_type']} | section: {metadata.get('section', '')}")
        print(f"   Chunk ID: {row['id']}")
        if "scenario_id" in metadata:
            start = metadata.get("chunk_event_time_start", metadata.get("event_time_start"))
            end = metadata.get("chunk_event_time_end", metadata.get("event_time_end"))
            print(f"   Synthetic scenario: {metadata['scenario_id']} | excerpt UTC: {start} -> {end}")
        if "snapshot_at" in metadata:
            print(f"   Snapshot: {metadata['snapshot_at']} | simulated: {metadata.get('snapshot_is_simulated')}")
        if metadata.get("document_context"):
            print(f"   Document context: {metadata['document_context']}")
        text = record["page_content"]
        print("   Evidence:" if full_text else "   Evidence excerpt (up to 1,000 characters):")
        print(text if full_text else text[:1000])
        if not full_text and len(text) > 1000:
            print("   [Display shortened; use --show-text for the complete evidence.]")


def show_candidate_rankings(rows: list[dict], records: list[dict], top_k: int) -> None:
    """Diagnostic display only; never changes candidates or their ranking."""
    print("\nAll merged candidates (dash = absent from that method's candidate list):")
    print("Fused | Dense | BM25 | Selected | Source / section")
    for rank, row in enumerate(rows, start=1):
        metadata = row["record"]["metadata"]
        print(f"{rank:5} | {str(row['dense_rank'] or '-'):>5} | "
              f"{str(row['bm25_rank'] or '-'):>4} | "
              f"{'yes' if rank <= top_k else 'no':>8} | "
              f"{metadata['source']} / {metadata.get('section', '')}")
    # A known section under investigation, not a mandatory retrieval result.
    target_ids = {
        record["id"] for record in records
        if record["metadata"]["source"] == "corpus/design/low-level-design.txt"
        and record["metadata"].get("section") == "3. VRRP"
    }
    if len(target_ids) != 1:
        print("\nVRRP diagnostic target not uniquely found in eligible local records.")
        return
    target_id = next(iter(target_ids))
    match = next(((rank, row) for rank, row in enumerate(rows, start=1)
                  if row["id"] == target_id), None)
    print("\nDiagnostic target: low-level-design.txt / 3. VRRP")
    if match is None:
        print("Absent from both returned candidate lists; its exact ranks are unknown.")
    else:
        rank, row = match
        print(f"Dense candidate rank: {row['dense_rank']} | BM25 candidate rank: {row['bm25_rank']}")
        print(f"Fused rank: {rank} | selected in top {top_k}: {rank <= top_k}")
        print(f"RRF contributions: dense={row['dense_contribution']:.6f}, BM25={row['bm25_contribution']:.6f}")
    print("Absence from a limited candidate list does not establish absence from the index.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--candidates", type=int, default=20, help="Candidates per method, 1 to 100.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of fused results, no larger than --candidates.")
    parser.add_argument("--show-text", action="store_true", help="Show full evidence instead of excerpts.")
    parser.add_argument("--show-candidates", action="store_true", help="Display all merged candidate ranks and diagnose the LLD VRRP section; no ranking changes.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preview", action="store_true", help="Local plan only (default).")
    modes.add_argument("--execute", action="store_true", help="Run both retrieval methods and fuse rankings.")
    args = parser.parse_args()
    question = args.question.strip()
    if not question or not network_tokens(question):
        parser.error("Question must contain searchable terms.")
    if not 1 <= args.candidates <= 100 or not 1 <= args.top_k <= args.candidates:
        parser.error("Use 1 <= top-k <= candidates <= 100.")
    try:
        records = read_chunks(args.input)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Cannot read export: {error}\n")
    scenario_ids = explicit_scenarios(question)
    eligible_records = scope_records(records, scenario_ids)
    known_scenarios = {
        record["metadata"].get("scenario_id") for record in records
        if record["metadata"]["document_type"] == "synthetic_log"
    }
    unknown_scenarios = sorted(set(scenario_ids) - known_scenarios)
    print(f"Mode: {'EXECUTE' if args.execute else 'PREVIEW'}")
    print(f"Question: {question}")
    print(f"Reference chunks: {len(records)}")
    print(f"Explicit scenario IDs: {scenario_ids or 'none (broad search)'}")
    print(f"Eligible chunks before ranking: {len(eligible_records)}/{len(records)}")
    if scenario_ids:
        print("Log scope: named scenarios only; all design/reference/configuration/topology evidence stays eligible.")
    if unknown_scenarios:
        print(f"No local log evidence for requested IDs: {unknown_scenarios}. Other scenarios will not substitute for them.")
    print(f"Dense search: {INDEX_NAME} / {NAMESPACE}; {EMBEDDING_MODEL}, {EMBEDDING_DIMENSIONS} dimensions")
    print(f"BM25: local LangChain retriever | tokens: {network_tokens(question)}")
    print(f"Candidates per method: {args.candidates} | final top-k: {args.top_k}")
    print(f"Fusion: equal weights, 1/({RRF_CONSTANT} + rank) per method; merge by chunk ID.")
    print("Raw BM25/cosine scores are not added. Exact score ties use chunk ID ordering.")
    print("Same scenario scope applies before both dense and BM25 ranking; RRF formula is unchanged.")
    print("Multiple literal IDs include each named scenario; descriptions alone do not trigger scoping.")
    print("No automatic rank boost, reranker or answer generator. Eligibility does not guarantee a top-k place.")
    print("All returned evidence must match the local export; RRF scores are not confidence.")
    if args.show_candidates:
        print("Diagnostic display enabled: all candidate ranks plus LLD VRRP target; ranking rules unchanged.")
    if not args.execute:
        print("\nPreview complete: no key requested, retrieval run, API calls made or files changed.")
        return

    stage = "building local BM25"
    try:
        lexical = bm25_results(
            build_bm25(eligible_records, args.candidates), eligible_records, question
        )
        # Check installed remote integration before requesting a secret.
        from langchain_pinecone import PineconeEmbeddings  # noqa: F401
        from pinecone import Pinecone  # noqa: F401
    except ImportError:
        parser.exit(1, "Missing retrieval dependencies. No remote calls made. Check the installed packages.\n")
    except ValueError as error:
        parser.exit(1, f"Local retrieval failed: {error}\n")

    print("\nBM25 candidates prepared. Execution sends the question to Pinecone; usage may be billed.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            api_key = getpass.getpass("Enter your existing Pinecone API key (hidden): ").strip()
    except (getpass.GetPassWarning, EOFError, KeyboardInterrupt):
        parser.exit(1, "\nKey entry cancelled or hidden input unavailable.\n")
    if not api_key:
        parser.exit(1, "No key entered. No remote calls made.\n")
    try:
        stage = "dense retrieval and evidence verification"
        dense = dense_candidates(
            eligible_records, question, api_key, args.candidates, scenario_ids
        )
        stage = "rank fusion"
        fused = fuse_rankings(dense, lexical)
        print(f"\nVerified candidates: dense={len(dense)}, BM25={len(lexical)}, unique merged={len(fused)}")
        if args.show_candidates:
            show_candidate_rankings(fused, eligible_records, args.top_k)
        for method, candidates in (("Dense", dense), ("BM25", lexical)):
            print(f"\n{method} first five candidates:")
            for rank, candidate in enumerate(candidates[:5], start=1):
                metadata = candidate["record"]["metadata"]
                print(f"  {rank}. {metadata['source']} | {metadata.get('section', '')}")
        show_results(fused, args.top_k, args.show_text)
        if not dense or not lexical:
            print("\nOne candidate list is empty: fusion could use only the available method.")
        print("\nNo answers generated, records modified or local files written.")
        print("Inspect required scenario and design evidence; fusion alone does not establish relevance.")
    except KeyboardInterrupt:
        parser.exit(1, f"\nInterrupted while {stage}. No records were written.\n")
    except ValueError as error:
        parser.exit(1, f"Validation stopped while {stage}: {error}\n")
    except Exception as error:
        parser.exit(1, f"Operation failed while {stage} ({type(error).__name__}). No records were written.\n")
    finally:
        del api_key


if __name__ == "__main__":
    main()
