"""Two offline product checks. No keys, provider calls or file writes.

Run: uv run python check_product.py
"""

from check_chunk_lengths import DEFAULT_EXPORT, read_chunks
from product_pipeline import retrieve_product


def main() -> None:
    records = read_chunks(DEFAULT_EXPORT)
    supported = retrieve_product(
        "In synthetic FAIL-08, what happened to TACACS+, syslog, SNMP and NTP during the management service failures?",
        records, evidence_limit=20)
    assert supported["route"] == "evidence_review", supported["route"]
    coverage = supported["package"]["scenario_coverage"][0]
    assert coverage["available_log_chunk_count"] == coverage["selected_log_chunk_count"] == 4
    evidence = supported["package"]["evidence"]
    assert evidence and all(item["text"] and item["source_line_ranges"] for item in evidence)
    assert len({item["citation_id"] for item in evidence}) == len(evidence)
    print("PASS 1: FAIL-08 local retrieval selects all four logs and preserves citation provenance.")

    # Hybrid mode with no key must still return the known-missing route locally.
    missing = retrieve_product("What happened in synthetic FAIL-99?", records, method="hybrid")
    assert missing["route"] == "missing_evidence", missing["route"]
    assert missing["package"]["evidence"] == []
    assert missing["package"]["missing_scenarios"] == ["FAIL-99"]
    print("PASS 2: missing FAIL-99 blocks generation eligibility without a provider key.")
    print("No provider calls made. These checks do not test model answer accuracy or browser layout.")


if __name__ == "__main__":
    main()
