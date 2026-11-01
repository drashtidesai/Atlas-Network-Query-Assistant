"""Product orchestration: reuse retrieval, evidence packaging and LangGraph routing."""

import re

from check_host_roles import ntp_host_issues
from prepare_evidence import build_evidence_package
from retrieve_bm25 import bm25_results, build_bm25, network_tokens
from retrieve_hybrid import dense_candidates, explicit_scenarios, fuse_rankings, scope_records
from route_evidence import build_routing_graph


def retrieve_product(question: str, records: list[dict], method: str = "local",
                     pinecone_key: str = "", candidates: int = 20,
                     evidence_limit: int = 12) -> dict:
    question = question.strip()
    if not network_tokens(question):
        raise ValueError("Enter a question containing searchable terms.")
    if method not in {"local", "hybrid"}:
        raise ValueError("Unknown retrieval method.")
    if not 1 <= evidence_limit <= candidates <= 100:
        raise ValueError("Invalid retrieval limits.")
    requested = explicit_scenarios(question)
    plan = build_evidence_package(question, records, None, evidence_limit, candidates)
    # Missing scenarios are known locally; no embedding request is needed.
    if plan["missing_scenarios"]:
        package = build_evidence_package(question, records, [], evidence_limit, candidates)
    else:
        eligible = scope_records(records, requested)
        lexical = bm25_results(build_bm25(eligible, candidates), eligible, question)
        if method == "hybrid":
            if not pinecone_key.strip():
                raise ValueError("Hybrid retrieval requires a Pinecone key.")
            dense = dense_candidates(eligible, question, pinecone_key, candidates, requested)
            ranked = fuse_rankings(dense, lexical)
        else:
            ranked = [{"id": row["id"], "record": row["record"],
                       "rrf_score": 1 / (60 + rank), "dense_rank": None,
                       "bm25_rank": rank} for rank, row in enumerate(lexical, 1)]
        package = build_evidence_package(question, records, ranked, evidence_limit, candidates)
    result = build_routing_graph().invoke({"package": package, "trace": ["retrieve_evidence"]})
    return {**result, "method": method}


def generate_product(package: dict, records: list[dict], api_key: str) -> dict:
    """One paid draft call; experimental model reviewer is not a product gate."""
    from answer_retrieved_evidence import answer_messages
    from chat_one_evidence import BASE_URL, MODEL
    from route_evidence import inspect_package
    from langchain_openai import ChatOpenAI

    if inspect_package({"package": package})["route"] != "evidence_review":
        raise ValueError("Evidence route does not permit generation.")
    if not api_key.strip():
        raise ValueError("Enter a Nebius key.")
    response = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=api_key,
                          temperature=0, max_tokens=900, timeout=90, max_retries=0).invoke(
                              answer_messages(package))
    if not isinstance(response.content, str) or not response.content.strip():
        raise ValueError("Empty model response.")
    draft = response.content
    used = set(re.findall(r"\[(S\d+)\]", draft))
    unknown = sorted(used - set(package["citations"]))
    issues = ntp_host_issues(draft, records)
    if not used:
        issues.append("Draft contains no recognized source citations.")
    if unknown:
        issues.append("Draft contains unknown source citations: " + ", ".join(unknown))
    if response.response_metadata.get("finish_reason") == "length":
        issues.append("Draft reached its output limit and may be incomplete.")
    return {"draft": draft, "issues": issues, "usage": response.usage_metadata or {},
            "citation_labels": sorted(used)}
