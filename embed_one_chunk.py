"""Lesson 5: preview or embed one checked chunk through LangChain.

Preview (default; no API calls or key prompt):
    uv run python embed_one_chunk.py

Execute (prompts for an existing Pinecone key; sends one text for embedding):
    uv run python embed_one_chunk.py --execute

Execution requires langchain-pinecone. No index is created, no vectors are
upserted, and no files are written. Pinecone embedding usage may be billed.
"""

import argparse
import getpass
import math
import warnings
from numbers import Real
from pathlib import Path

from check_chunk_lengths import DEFAULT_EXPORT, embedding_text, read_chunks
from rag_settings import EMBEDDING_MODEL as MODEL, EMBEDDING_DIMENSIONS as DIMENSIONS


SOURCE = "corpus/synthetic-logs/test-08-management-service-failure.txt"
SECTION = "08A-08C TACACS and console"


def select_chunk(records: list[dict]) -> dict:
    """Select the service chunk we measured at 1,403 preliminary tokens."""
    matches = [
        record for record in records
        if record["metadata"]["source"] == SOURCE
        and record["metadata"].get("section") == SECTION
    ]
    if len(matches) != 1:
        raise ValueError("Expected exactly one FAIL-08 TACACS/console chunk. Check the latest export.")
    record = matches[0]
    if (
        record["metadata"].get("document_type") != "synthetic_log"
        or record["metadata"].get("scenario_id") != "FAIL-08"
        or record["metadata"].get("evidence_is_simulated") is not True
    ):
        raise ValueError("Selected chunk is missing its synthetic scenario identity.")
    return record


def validate_vector(vectors: list[list[float]]) -> list[float]:
    """Check shape and numeric integrity, not retrieval accuracy."""
    if len(vectors) != 1:
        raise ValueError("Expected one embedding for one input text.")
    vector = vectors[0]
    if len(vector) != DIMENSIONS:
        raise ValueError(f"Expected {DIMENSIONS} dimensions, received {len(vector)}.")
    if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in vector):
        raise ValueError("Vector must contain only finite numbers.")
    if not any(value != 0 for value in vector):
        raise ValueError("Received an all-zero vector.")
    return vector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_EXPORT, help="Existing checked JSONL export.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true", help="Preview only (the default).")
    mode.add_argument("--execute", action="store_true", help="Prompt for a key and embed one text via Pinecone.")
    parser.add_argument("--show-text", action="store_true", help="Display the full prepared input instead of a short excerpt.")
    args = parser.parse_args()

    try:
        record = select_chunk(read_chunks(args.input))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Cannot select the chunk: {error}\n")

    text = embedding_text(record)
    metadata = record["metadata"]
    print(f"Mode: {'EXECUTE' if args.execute else 'PREVIEW'}")
    print(f"Model: {MODEL} | requested dimensions: {DIMENSIONS}")
    print("Input type: passage | truncation: NONE (reject oversized input)")
    print(f"Source: {metadata['source']}")
    print(f"Section: {metadata['section']}")
    print(f"Chunk ID: {record['id']}")
    print(f"Precise source ranges: {metadata.get('source_line_ranges')}")
    print(f"Prepared input characters: {len(text)} (not a token count)")
    print("\nPrepared input" + (":" if args.show_text else " excerpt (first 600 characters):"))
    print(text if args.show_text else text[:600])

    if not args.execute:
        print("\nPreview complete: no key requested, API calls made, or files changed.")
        return

    # Import only in execution mode: preview works before this is installed.
    try:
        from langchain_pinecone import PineconeEmbeddings
    except ImportError:
        parser.exit(1, "Execution needs langchain-pinecone. Install it with: uv add langchain-pinecone\nNo API call was made.\n")

    print("\nExecution sends this one prepared text to Pinecone; usage may be billed.")
    print("The key is used for this process only; this script does not save it.")
    try:
        # Refuse getpass's visible-input fallback if a terminal cannot hide input.
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            api_key = getpass.getpass("Enter your existing Pinecone API key (hidden): ").strip()
    except (getpass.GetPassWarning, EOFError, KeyboardInterrupt):
        parser.exit(1, "\nKey entry cancelled or hidden input unavailable. Use an interactive terminal.\n")
    if not api_key:
        parser.exit(1, "No key entered. No API call was made.\n")

    try:
        # Construction may query Pinecone's model catalog before embedding.
        embeddings = PineconeEmbeddings(
            model=MODEL,
            dimension=DIMENSIONS,
            pinecone_api_key=api_key,
            document_params={
                "input_type": "passage",
                "dimension": DIMENSIONS,
                "truncate": "NONE",
            },
            query_params={
                "input_type": "query",
                "dimension": DIMENSIONS,
                "truncate": "NONE",
            },
        )
        vectors = embeddings.embed_documents([text])
    except Exception as error:
        # Avoid dumping exception payloads, which can contain request secrets.
        parser.exit(1, f"Pinecone/LangChain request failed ({type(error).__name__}).\nCheck credentials, model access, connectivity and package compatibility. No index or files were written.\n")
    finally:
        del api_key

    try:
        vector = validate_vector(vectors)
    except ValueError as error:
        parser.exit(1, f"Embedding validation failed: {error}\n")

    print(f"\nReceived one embedding with {len(vector)} finite numeric values.")
    print(f"First five values: {[round(float(value), 6) for value in vector[:5]]}")
    print("This input was accepted with truncate=NONE; other chunks are not server-validated yet.")
    print("Vector shape validation does not establish retrieval quality.")
    print("No index created, vectors upserted, or files written. Vector remains in memory only.")


if __name__ == "__main__":
    main()
