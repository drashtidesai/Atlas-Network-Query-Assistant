"""Lesson 14: real hybrid retrieval inside a LangGraph routing workflow.

Preview: uv run python run_rag_routing.py
Execute: uv run python run_rag_routing.py --execute

Execution prompts for a key, embeds the question, queries Pinecone read-only,
builds an evidence package and routes it. No answering model is used. No local
files or remote records are written. Embedding/search usage may be billed.
"""

import argparse
import getpass
import json
import warnings
from pathlib import Path

from check_chunk_lengths import DEFAULT_EXPORT, read_chunks
from prepare_evidence import build_evidence_package, print_summary
from rag_settings import INDEX_NAME, NAMESPACE
from retrieve_bm25 import bm25_results, build_bm25, network_tokens
from retrieve_dense import DEFAULT_QUESTION
from retrieve_hybrid import dense_candidates, explicit_scenarios, fuse_rankings, scope_records
from route_evidence import (
    ROUTE_MESSAGES, RoutingState, choose_next_node, inspect_package, report_outcome,
)


class LiveRoutingState(RoutingState):
    """Adds the user's question to the existing evidence-routing state."""
    question: str


def build_live_graph(records: list[dict], api_key: str, candidates: int, top_k: int):
    """Keep the credential in a closure, not in graph state or printed output."""
    from langgraph.graph import END, START, StateGraph

    def retrieve_evidence(state: LiveRoutingState) -> dict:
        question = state["question"]
        requested = explicit_scenarios(question)
        eligible = scope_records(records, requested)
        print("Node retrieve_evidence: building local BM25 candidates...", flush=True)
        lexical = bm25_results(build_bm25(eligible, candidates), eligible, question)
        print("Node retrieve_evidence: embedding the question and searching Pinecone...", flush=True)
        dense = dense_candidates(eligible, question, api_key, candidates, requested)
        fused = fuse_rankings(dense, lexical)
        package = build_evidence_package(question, records, fused, top_k, candidates)
        print(f"Node retrieve_evidence: package built with {len(package['evidence'])} evidence items.", flush=True)
        return {
            "package": package,
            "trace": state.get("trace", []) + ["retrieve_evidence"],
        }

    builder = StateGraph(LiveRoutingState)
    builder.add_node("retrieve_evidence", retrieve_evidence)
    builder.add_node("inspect_package", inspect_package)
    builder.add_edge(START, "retrieve_evidence")
    builder.add_edge("retrieve_evidence", "inspect_package")
    for route in ROUTE_MESSAGES:
        builder.add_node(route, report_outcome)
        builder.add_edge(route, END)
    builder.add_conditional_edges(
        "inspect_package", choose_next_node,
        {route: route for route in ROUTE_MESSAGES},
    )
    return builder.compile()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--show-evidence", action="store_true", help="Print the full package after execution.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preview", action="store_true", help="Local plan only (default).")
    modes.add_argument("--execute", action="store_true", help="Run real retrieval and LangGraph routing.")
    args = parser.parse_args()
    question = args.question.strip()
    if not question or not network_tokens(question):
        parser.error("Question must contain searchable terms.")
    if not 1 <= args.top_k <= args.candidates <= 100:
        parser.error("Use 1 <= top-k <= candidates <= 100.")
    try:
        records = read_chunks(args.input)
        plan = build_evidence_package(question, records, None, args.top_k, args.candidates)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Cannot prepare the workflow: {error}\n")

    print(f"Mode: {'EXECUTE (real retrieval + routing)' if args.execute else 'PREVIEW'}")
    print(f"Question: {question}")
    print(f"Reference chunks: {len(records)} | eligible chunks: {plan['retrieval']['eligible_chunk_count']}")
    print(f"Requested scenarios: {plan['requested_scenarios']}")
    print(f"Missing from local corpus: {plan['missing_scenarios']}")
    print(f"Dense destination: {INDEX_NAME} / {NAMESPACE}; BM25 stays local.")
    print(f"Candidates per method: {args.candidates} | evidence limit: {args.top_k}")
    print("Graph: START -> retrieve_evidence -> inspect_package -> conditional outcome -> END")
    print("Evidence comes from real retrieval in execution mode, not hand-selected examples.")
    print("evidence_review means further assessment is needed; no answer generation is enabled.")
    if not args.execute:
        print("\nPreview complete: graph not run, no outcome inferred, key requested or API calls made.")
        return

    try:
        from langgraph.graph import StateGraph  # noqa: F401
        from langchain_pinecone import PineconeEmbeddings  # noqa: F401
        from langchain_community.retrievers import BM25Retriever  # noqa: F401
        from rank_bm25 import BM25Okapi  # noqa: F401
    except ImportError:
        parser.exit(1, "A required graph/retrieval dependency is unavailable. No remote calls made.\n")
    print("\nExecution uses Pinecone embedding/search quota; the key is not saved by this script.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            api_key = getpass.getpass("Enter your existing Pinecone API key (hidden): ").strip()
    except (getpass.GetPassWarning, EOFError, KeyboardInterrupt):
        parser.exit(1, "\nKey entry cancelled or hidden input unavailable.\n")
    if not api_key:
        parser.exit(1, "No key entered. No remote calls made.\n")

    try:
        graph = build_live_graph(records, api_key, args.candidates, args.top_k)
        result = graph.invoke({"question": question, "trace": []})
        expected_trace = ["retrieve_evidence", "inspect_package", result["route"]]
        if result["trace"] != expected_trace:
            raise ValueError("Unexpected workflow path.")
        print("\nActual graph path: " + " -> ".join(["START"] + result["trace"] + ["END"]))
        print(f"Route: {result['route']}")
        print(f"Routing message: {result['message']}")
        print_summary(result["package"])
        if args.show_evidence:
            print(json.dumps(result["package"], indent=2, ensure_ascii=False, allow_nan=False))
        print("\nEvidence package stayed in graph state through retrieval, inspection and outcome.")
        print("No answer generated, records modified or local files written.")
    except KeyboardInterrupt:
        parser.exit(1, "\nWorkflow interrupted. No records or files were written.\n")
    except ValueError as error:
        parser.exit(1, f"Workflow validation failed: {error}\n")
    except Exception as error:
        parser.exit(1, f"Workflow failed ({type(error).__name__}). No records or files were written.\n")
    finally:
        del api_key


if __name__ == "__main__":
    main()
