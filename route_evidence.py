"""Lesson 13: first LangGraph routing checkpoint, using local examples only.

Preview: uv run python route_evidence.py
Run graph checks: uv run python route_evidence.py --execute
One case: uv run python route_evidence.py --execute --case missing

Requires langgraph for execution, not preview. Fixtures use real local source
chunks but hand-selected evidence, NOT actual retrieval results. No Pinecone,
embedding, answering-model or telemetry calls are made by this script.
"""

import argparse
from pathlib import Path
from typing import TypedDict

from check_chunk_lengths import DEFAULT_EXPORT, read_chunks
from prepare_evidence import build_evidence_package


class RoutingState(TypedDict):
    """The shared information carried between graph nodes."""
    package: dict
    route: str
    message: str
    trace: list[str]


ROUTE_MESSAGES = {
    "retrieval_not_run": "Retrieval has not completed. Obtain evidence before preparing an answer.",
    "missing_evidence": "Requested scenario logs are missing. Report that limitation; design guidance cannot establish what happened in that scenario.",
    "retrieval_gap": "Some available scenario log chunks were not selected. Review retrieval or narrow the question before answering about the whole scenario.",
    "no_evidence": "No evidence items were selected. Report unavailable supporting evidence; do not invent an answer.",
    "evidence_review": "Evidence is present and no scenario-selection gap was reported. Next: assess relevance and sufficiency; answer generation is not enabled.",
}


def inspect_package(state: RoutingState) -> dict:
    """Choose a route from objective package status, not similarity scores."""
    package = state["package"]
    if package.get("schema_version") != "1.0":
        raise ValueError("Expected evidence-package schema 1.0.")
    status = package.get("retrieval_status")
    if status not in {"not_run", "completed"}:
        raise ValueError("Unknown retrieval status; refusing to route to evidence review.")
    if not isinstance(package.get("evidence"), list):
        raise ValueError("Evidence must be a list.")
    if not isinstance(package.get("missing_scenarios"), list):
        raise ValueError("Missing-scenario status must be a list.")

    if status == "not_run":
        route = "retrieval_not_run"
    else:
        for field in ("scenarios_not_retrieved", "partially_retrieved_scenarios"):
            if not isinstance(package.get(field), list):
                raise ValueError(f"Completed retrieval needs an explicit list for {field}.")
        # Priority: known missing logs, selection gaps, empty results, review.
        if package["missing_scenarios"]:
            route = "missing_evidence"
        elif package["scenarios_not_retrieved"] or package["partially_retrieved_scenarios"]:
            route = "retrieval_gap"
        elif not package["evidence"]:
            route = "no_evidence"
        else:
            route = "evidence_review"
    return {"route": route, "trace": state.get("trace", []) + ["inspect_package"]}


def choose_next_node(state: RoutingState) -> str:
    """This function is the conditional edge: it selects a node name."""
    return state["route"]


def report_outcome(state: RoutingState) -> dict:
    """A deterministic terminal node; it does not call an answering model."""
    route = state["route"]
    return {
        "message": ROUTE_MESSAGES[route],
        "trace": state["trace"] + [route],
    }


def build_routing_graph():
    """Compile a graph without persistence, tracing callbacks or external tools."""
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(RoutingState)
    builder.add_node("inspect_package", inspect_package)
    builder.add_edge(START, "inspect_package")
    for route in ROUTE_MESSAGES:
        builder.add_node(route, report_outcome)
        builder.add_edge(route, END)
    builder.add_conditional_edges(
        "inspect_package", choose_next_node,
        {route: route for route in ROUTE_MESSAGES},
    )
    return builder.compile()


def local_examples(records: list[dict]) -> dict:
    """Create labelled routing fixtures without invoking either retriever."""
    def find(source: str, section: str | None = None) -> dict:
        return next(record for record in records if record["metadata"]["source"] == source
                    and (section is None or record["metadata"].get("section") == section))

    design = find("corpus/design/low-level-design.txt", "3. VRRP")
    log = find("corpus/synthetic-logs/test-01-wan-a-silent-loss.txt")
    partial_log = find("corpus/synthetic-logs/test-08-management-service-failure.txt", "08F NTP")

    def fixture(question: str, selected: list[dict] | None) -> dict:
        rows = None if selected is None else [
            {"id": record["id"], "record": record, "rrf_score": 0.0,
             "dense_rank": None, "bm25_rank": None}
            for record in selected
        ]
        package = build_evidence_package(question, records, rows, 5, 20)
        package["fixture_only"] = True
        package["retrieval"]["method"] = "hand_selected_local_routing_fixture"
        return package

    return {
        "missing": (fixture("What happened in synthetic FAIL-99?", [design]), "missing_evidence"),
        "gap": (fixture("What happened in synthetic FAIL-01?", [design]), "retrieval_gap"),
        "partial": (fixture("Summarise all management subtests in FAIL-08.", [partial_log]), "retrieval_gap"),
        "not-run": (fixture("What happened in synthetic FAIL-01?", None), "retrieval_not_run"),
        "empty": (fixture("What does the evidence say about VRRP?", []), "no_evidence"),
        "review": (fixture("What happened in synthetic FAIL-01?", [design, log]), "evidence_review"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--case", choices=["all", "missing", "gap", "partial", "not-run", "empty", "review"], default="all")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preview", action="store_true", help="Show routing examples without running LangGraph (default).")
    modes.add_argument("--execute", action="store_true", help="Run local LangGraph fixtures; no external calls.")
    args = parser.parse_args()
    try:
        examples = local_examples(read_chunks(args.input))
    except (OSError, ValueError, StopIteration) as error:
        parser.exit(1, f"Cannot prepare local routing examples ({type(error).__name__}). Check the current export.\n")
    selected = examples if args.case == "all" else {args.case: examples[args.case]}
    print(f"Mode: {'EXECUTE (local routing only)' if args.execute else 'PREVIEW'}")
    print("LOCAL FIXTURES: real source chunks, hand-selected evidence; no retrieval performed.")
    print("Graph: START -> inspect_package -> conditional outcome node -> END")
    for name, (package, expected) in selected.items():
        print(f"  {name}: expected={expected}; missing={package['missing_scenarios']}; missed={package['scenarios_not_retrieved']}; partial={package['partially_retrieved_scenarios']}; evidence_items={len(package['evidence'])}")
    if not args.execute:
        print("\nPreview complete: no LangGraph execution, key prompts, API calls or file writes.")
        return

    try:
        graph = build_routing_graph()
    except ImportError:
        parser.exit(1, "LangGraph is not installed. Install with: uv add langgraph\nNo graph execution or API calls occurred.\n")
    for name, (package, expected) in selected.items():
        result = graph.invoke({"package": package, "trace": []})
        if result["route"] != expected or result["trace"] != ["inspect_package", expected]:
            raise ValueError(f"Unexpected graph path for local example: {name}")
        print(f"\n{name}: PASS | {' -> '.join(['START'] + result['trace'] + ['END'])}")
        print(f"  {result['message']}")
    print(f"\nPassed {len(selected)} local routing checks. Real retrieval integration is not tested here.")
    print("No answers generated, API keys requested, external calls made or files changed.")


if __name__ == "__main__":
    main()
