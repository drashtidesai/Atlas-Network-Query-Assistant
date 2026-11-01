"""Preview or execute 15 retrieval cases; never calls the generation model.

uv run python evaluate_qa15.py
uv run python evaluate_qa15.py --execute
uv run python evaluate_qa15.py --execute --method hybrid
Execution writes a timestamped JSON trace and text evaluation report under evaluation/.
Hybrid execution prompts for a Pinecone key and consumes embedding/search quota.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from check_chunk_lengths import DEFAULT_EXPORT, read_chunks
from product_pipeline import retrieve_product


ROOT = Path(__file__).resolve().parent


def matches(item: dict, anchor: dict) -> bool:
    metadata = item.get("source_metadata", item.get("metadata", {}))
    return all(metadata.get(key) == value for key, value in anchor.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--method", choices=["local", "hybrid"], default="local")
    args = parser.parse_args()
    fixture = ROOT / "evaluation" / "qa15.json"
    cases = json.loads(fixture.read_text(encoding="utf-8"))["cases"]
    records = read_chunks(DEFAULT_EXPORT)
    if len(cases) != 15 or len({case["id"] for case in cases}) != 15:
        parser.error("Expected 15 unique test cases.")
    # Validate anchor existence before keys, retrieval or output writes.
    for case in cases:
        for group in case["anchors"]:
            if not any(matches(record, anchor) for record in records for anchor in group):
                parser.error(f"Unresolved expected evidence anchor in {case['id']}: {group}")
        print(f"{case['id']}: {case['question']}")
    print(f"\nMethod={args.method}; candidates per method=20; selected evidence=20.")
    if not args.execute:
        print("Preview only. All anchors resolved. No keys, retrieval calls or file writes.")
        return

    key = ""
    if args.method == "hybrid":
        from answer_retrieved_evidence import hidden_key
        print("Hybrid run makes up to 14 embedding/search operations; provider usage may apply.")
        key = hidden_key("Pinecone")
    results = []
    try:
        for case in cases:
            row = {**case, "error": None, "hit5": None, "rr20": None,
                   "anchor_coverage20": None, "route_correct": False, "retrieved": []}
            try:
                result = retrieve_product(case["question"], records, args.method, key,
                                          candidates=20, evidence_limit=20)
                items = result["package"]["evidence"]
                groups = case["anchors"]
                ranks = [rank for rank, item in enumerate(items, 1)
                         if any(matches(item, anchor) for group in groups for anchor in group)]
                if groups:
                    row["hit5"] = int(bool(ranks) and min(ranks) <= 5)
                    row["rr20"] = 1 / min(ranks) if ranks else 0
                    row["anchor_coverage20"] = sum(any(matches(item, anchor)
                        for item in items for anchor in group) for group in groups) / len(groups)
                row["route"] = result["route"]
                row["route_correct"] = result["route"] == case["route"]
                row["expected_route"] = case["route"]
                row["retrieved"] = [{key: item[key] for key in (
                    "citation_id", "chunk_id", "source", "section", "source_line_ranges", "text")}
                    for item in items]
                row["failures"] = []
                if groups and not ranks:
                    row["failures"].append("Expected anchor evidence absent from selected top 20.")
                elif groups and min(ranks) > 5:
                    row["failures"].append("Expected evidence found, but ranked below top 5.")
                if groups and row["anchor_coverage20"] < 1:
                    row["failures"].append("Some required evidence facets were not retrieved.")
                if not row["route_correct"]:
                    row["failures"].append("Actual route differs from expected coverage route.")
            except Exception as error:
                # No provider payloads or credentials in saved artifacts.
                row["error"] = type(error).__name__
                row["failures"] = ["Execution error; inspect connectivity/dependencies separately."]
                if case["anchors"]:
                    row.update(hit5=0, rr20=0, anchor_coverage20=0)
            results.append(row)
            print(f"{case['id']}: hit5={row['hit5']} coverage20={row['anchor_coverage20']} route_ok={row['route_correct']} error={row['error']}")
    finally:
        del key

    scored = [row for row in results if row["anchors"]]
    averages = {metric: sum(row[metric] for row in scored) / len(scored)
                for metric in ("hit5", "rr20", "anchor_coverage20")}
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    base = ROOT / "evaluation" / f"qa15-{args.method}-{stamp}"
    artifact = {"run_utc": now.isoformat(), "method": args.method,
                "candidates": 20, "evidence_limit": 20,
                "corpus_sha256": hashlib.sha256(DEFAULT_EXPORT.read_bytes()).hexdigest(),
                "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                "averages": averages, "results": results}
    report = ["ATLAS NETWORK QUERY ASSISTANT - 15-QUESTION EVALUATION", "",
        f"Run UTC: {artifact['run_utc']}", f"Retrieval: {args.method}; candidates=20; selected evidence=20.",
        "Generation was not executed. This is a retrieval/routing evaluation, not answer-accuracy scoring.", "",
        "METHOD AND SCORE DEFINITIONS",
        "14 answerable cases plus one missing-scenario case; fixed authored anchors validated against the corpus.",
        "Alternatives within an anchor group are interchangeable; separate groups represent required evidence facets.",
        "Hit@5: at least one expected anchor appears in the first five selected results (0 or 1).",
        "MRR@20: mean reciprocal rank of the first expected anchor, or zero when absent.",
        "Anchor coverage@20: fraction of required evidence groups represented in the selected results.",
        "Scores average 14 answerable cases; missing-scenario retrieval metrics are N/A. Errors count as zero.",
        "Route accuracy uses all 15 cases. Anchors are not exhaustive relevance labels; precision is not measured.",
        "This small authored development set is not a held-out benchmark. Scores do not establish semantic answer correctness.", "",
        "MEASURED SUMMARY",
        f"Hit@5: {averages['hit5']:.3f}", f"MRR@20: {averages['rr20']:.3f}",
        f"Mean anchor coverage@20: {averages['anchor_coverage20']:.3f}",
        f"Route accuracy: {sum(row['route_correct'] for row in results)}/15",
        f"Execution errors: {sum(row['error'] is not None for row in results)}", "",
        "PER-QUESTION RESULTS"]
    for row in results:
        report += ["", f"{row['id']}: {row['question']}", f"Expected answer: {row['expected_answer']}",
                   f"Hit@5={row['hit5']} | RR@20={row['rr20']} | anchor coverage@20={row['anchor_coverage20']} | route correct={row['route_correct']}",
                   "Failure analysis: " + ("; ".join(row["failures"]) or "No failure under the anchor/routing criteria; answer accuracy not assessed.")]
    report += ["", "PREVIOUS MANUAL ANSWER OBSERVATIONS (SEPARATE FROM THIS RUN)",
        "London gateway draft matched documented VLAN 120 and VIP 10.20.20.1.",
        "FAIL-01 drafts sometimes confused the routed interrouter link with the interswitch link.",
        "FAIL-06 at evidence limit 10 confused same-subnet bridging with inter-VLAN routing and overstated lost connectivity.",
        "FAIL-08 drafts misplaced the local NTP application on DXB-R02 and sometimes merged TACACS fallback and console subtests.",
        "A model reviewer falsely accepted the wrong NTP host in a fixed two-case test (one correct acceptance, one false acceptance).",
        "A narrow topology-based wording check caught that known NTP error; it is not general factual validation.",
        "20/15/10 evidence-limit comparison was exploratory, not a controlled repeated quality experiment.", "",
        "LEARNINGS AND NEXT STEPS",
        "Separate retrieval ranking, evidence coverage, routing and answer correctness.",
        "Use low top-5 hits to investigate ranking; incomplete facet coverage to investigate selection; wrong routes to investigate gating.",
        "Review citations at claim level. Test paraphrases and negatives; expand independent labels before claiming general quality.",
        "Re-run with hybrid mode separately to compare measured retrieval on the same fixtures; this consumes Pinecone quota.",
        "Corpus SHA256: " + artifact["corpus_sha256"], "Fixture SHA256: " + artifact["fixture_sha256"]]
    base.with_suffix(".json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    base.with_suffix(".txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nReport: {base.with_suffix('.txt')}\nTrace: {base.with_suffix('.json')}")
    print("No generation calls. Corpus unchanged. Report and trace written under evaluation/.")


if __name__ == "__main__":
    main()
