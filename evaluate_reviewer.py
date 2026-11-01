"""Preview fixed NTP reviewer cases; --execute makes one Nebius review call per case.

No retrieval, draft generation, corpus changes or result file writes.
    uv run python evaluate_reviewer.py
    uv run python evaluate_reviewer.py --execute
"""

import argparse
import json
from pathlib import Path

from check_chunk_lengths import DEFAULT_EXPORT, read_chunks
from check_host_roles import ntp_host_issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        fixture_path = Path(__file__).resolve().parent / "evaluation" / "reviewer-cases.json"
        cases = json.loads(fixture_path.read_text(encoding="utf-8"))["cases"]
        records = read_chunks(args.input)
        matches = [record for record in records
                   if record["metadata"].get("scenario_id") == "FAIL-08"
                   and record["id"].endswith(":synthetic-service:ntp")]
        if len(matches) != 1:
            raise ValueError("Expected exactly one FAIL-08 NTP source chunk.")
        source = matches[0]
        # This isolated reviewer test deliberately supplies fixed evidence, without
        # claiming that retrieval ran or passing through whole-scenario routing.
        package = {
            "question": "In synthetic FAIL-08, where did the NTP application stop and what did DXB-R02 record?",
            "evidence": [{"citation_id": "S1", "chunk_id": source["id"],
                          "text": source["page_content"], "source_metadata": source["metadata"]}],
        }
        print("Fixed reviewer evaluation: NTP host distinction only.")
        print(f"Evidence S1: {source['metadata']['source']} / {source['metadata'].get('section', '')}")
        for case in cases:
            print(f"\n{case['id']} | expected={case['expected']}\n{case['draft']}")
            issues = ntp_host_issues(case["draft"], records)
            print(f"Local host-role check: {'flagged' if issues else 'no matching issue'}")
            for issue in issues:
                print(f"  {issue}")
        if not args.execute:
            print("\nPreview only: no key prompt, model calls or file writes.")
            print(f"--execute makes {len(cases)} paid review calls; no drafts are generated.")
            return

        from answer_retrieved_evidence import hidden_key, review_draft
        from chat_one_evidence import BASE_URL, MODEL
        from langchain_openai import ChatOpenAI

        print(f"\nExecuting {len(cases)} paid Nebius review calls; retries 0.")
        key = hidden_key("Nebius")
        passed = 0
        inconclusive = 0
        try:
            chat = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=key,
                              temperature=0, max_tokens=3000, timeout=90, max_retries=0)
            for case in cases:
                review, usage = review_draft(chat, package, case["draft"])
                status = review["status"]
                if status == "invalid_review":
                    outcome = "INCONCLUSIVE"
                    inconclusive += 1
                else:
                    # A formatting failure is not successful factual-error detection.
                    detected = any(item["verdict"] == "unsupported" for item in review["claims"])
                    correct = (status == "passed_model_review" if case["expected"] == "accept"
                               else status == "needs_revision" and detected)
                    outcome = "PASS" if correct else "FAIL"
                    passed += int(correct)
                print(f"\n{case['id']}: {outcome} | review={status}")
                if review.get("validation_reason"):
                    print(f"Reason: {review['validation_reason']} | entry={review.get('claim_number')}")
                for item in review["claims"]:
                    print(f"  {item['sentence_id']} {item['verdict']}: {item['reason']}")
                print(f"Reported review usage: {usage}")
        finally:
            del key
        print(f"\nPassed: {passed}/{len(cases)} | inconclusive: {inconclusive}")
        print("One pair of cases does not establish general reviewer reliability.")
        print("No corpus changes or file writes.")
    except (Exception, KeyboardInterrupt) as error:
        parser.exit(1, f"Evaluation stopped ({type(error).__name__}). No files written; already-sent calls may incur usage.\n")


if __name__ == "__main__":
    main()
