"""Lesson 7: preview or embed, upload and fetch back one chunk.

Preview: uv run python upload_one_chunk.py
Execute: uv run python upload_one_chunk.py --execute

Execution prompts for an existing Pinecone key and may incur inference/database
usage. It never creates/deletes indexes or writes local files. A matching
existing record is verified and reused; a conflicting record is not overwritten.
"""

import argparse
import getpass
import hashlib
import json
import math
import time
import warnings
from pathlib import Path

from check_chunk_lengths import DEFAULT_EXPORT, embedding_text, read_chunks
from embed_one_chunk import select_chunk, validate_vector
from rag_settings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, INDEX_NAME, NAMESPACE
from setup_pinecone_index import check_configuration


def storage_metadata(record: dict, text: str) -> dict:
    """Preserve source metadata, original evidence text and embedding provenance."""
    metadata = record["metadata"].copy()
    additions = {
        # 'text' is also the default text field for LangChain's Pinecone store.
        "text": record["page_content"],
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSIONS,
        "embedding_input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "embedding_input_format": "source-labels-context-evidence-v1",
    }
    if metadata.keys() & additions.keys():
        raise ValueError("Source metadata conflicts with a storage/provenance field.")
    metadata.update(additions)
    for key, value in metadata.items():
        if not isinstance(key, str) or key.startswith("$"):
            raise ValueError("Pinecone metadata keys must be strings without a leading $.")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Metadata numbers must be finite.")
        if not (
            isinstance(value, (str, bool, int, float))
            or isinstance(value, list) and all(isinstance(item, str) for item in value)
        ):
            raise ValueError(f"Unsupported Pinecone metadata type for {key}.")
    if len(json.dumps(metadata, ensure_ascii=False).encode("utf-8")) > 40_000:
        raise ValueError("Record metadata exceeds our conservative 40,000-byte check.")
    return metadata


def verify_record(saved: dict, chunk_id: str, metadata: dict, expected_vector=None) -> None:
    """Check the fetched record's identity, source evidence and vector."""
    if saved.get("id") != chunk_id:
        raise ValueError("Fetched record has the wrong ID.")
    if saved.get("metadata") != metadata:
        raise ValueError("Stored text or metadata differs; record was not overwritten.")
    vector = validate_vector([saved.get("values", [])])
    if expected_vector is not None and any(
        not math.isclose(float(actual), float(expected), rel_tol=1e-5, abs_tol=1e-6)
        for actual, expected in zip(vector, expected_vector)
    ):
        raise ValueError("Fetched vector differs from the vector submitted for upload.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_EXPORT)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preview", action="store_true", help="Show the plan only (default).")
    modes.add_argument("--execute", action="store_true", help="Embed/upload one record and verify it remotely.")
    args = parser.parse_args()

    try:
        record = select_chunk(read_chunks(args.input))
        text = embedding_text(record)
        metadata = storage_metadata(record, text)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Cannot prepare record: {error}\n")

    chunk_id = record["id"]
    print(f"Mode: {'EXECUTE' if args.execute else 'PREVIEW'}")
    print(f"Index: {INDEX_NAME} | namespace: {NAMESPACE}")
    print(f"Model: {EMBEDDING_MODEL} | dimensions: {EMBEDDING_DIMENSIONS}")
    print("Input type: passage | truncation: NONE")
    print(f"Source: {metadata['source']}")
    print(f"Section: {metadata['section']}")
    print(f"Precise source ranges: {metadata.get('source_line_ranges')}")
    print(f"Chunk ID: {chunk_id}")
    print(f"Prepared embedding input: {len(text)} characters (not tokens)")
    print(f"Stored original evidence: {len(record['page_content'])} characters")
    print("All source metadata and an embedding-input fingerprint will accompany the vector.")
    print("Existing matching ID: verify/reuse. Conflicting ID: stop without overwriting.")

    if not args.execute:
        print("\nPreview complete: no key requested, API calls made or local files changed.")
        return

    try:
        from langchain_pinecone import PineconeEmbeddings
        from pinecone import Pinecone
    except ImportError:
        parser.exit(1, "Missing Pinecone/LangChain integration. Check the installed dependencies.\n")

    print("\nExecution sends one prepared text for embedding and stores its vector, text and metadata.")
    print("Usage may be billed. The key is used in memory only and is not saved by this script.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            api_key = getpass.getpass("Enter your existing Pinecone API key (hidden): ").strip()
    except (getpass.GetPassWarning, EOFError, KeyboardInterrupt):
        parser.exit(1, "\nKey entry cancelled or hidden input unavailable. Use an interactive terminal.\n")
    if not api_key:
        parser.exit(1, "No key entered. No API call was made.\n")

    stage = "checking index"
    try:
        client = Pinecone(api_key=api_key)
        description = client.describe_index(INDEX_NAME).to_dict()
        check_configuration(description)
        if description.get("status", {}).get("ready") is not True:
            raise ValueError("Index is not ready; no embedding/upload attempted.")
        index = client.Index(host=description["host"])

        stage = "checking existing record"
        # SDK 7.3 returns a FetchResponse dataclass with a .vectors attribute.
        existing = index.fetch(ids=[chunk_id], namespace=NAMESPACE).vectors
        if chunk_id in existing:
            verify_record(existing[chunk_id].to_dict(), chunk_id, metadata)
            print("\nMatching record already exists. ID, text, metadata and vector shape verified.")
            print("No new embedding requested or record uploaded.")
            return

        stage = "embedding one input"
        embeddings = PineconeEmbeddings(
            model=EMBEDDING_MODEL,
            dimension=EMBEDDING_DIMENSIONS,
            pinecone_api_key=api_key,
            document_params={"input_type": "passage", "dimension": EMBEDDING_DIMENSIONS, "truncate": "NONE"},
            query_params={"input_type": "query", "dimension": EMBEDDING_DIMENSIONS, "truncate": "NONE"},
        )
        vector = validate_vector(embeddings.embed_documents([text]))
        print(f"Received one {len(vector)}-dimensional embedding with truncation disabled.")

        stage = "upserting one record"
        response = index.upsert(
            vectors=[{"id": chunk_id, "values": vector, "metadata": metadata}],
            namespace=NAMESPACE,
        )
        if response.upserted_count != 1:
            raise ValueError("Upsert did not acknowledge exactly one record.")
        print("One record acknowledged. Fetching it back for verification...")

        stage = "fetching uploaded record"
        # Allow brief propagation delay; retry only fetch, never the embedding.
        for attempt in range(6):
            fetched = index.fetch(ids=[chunk_id], namespace=NAMESPACE).vectors
            if chunk_id in fetched:
                verify_record(fetched[chunk_id].to_dict(), chunk_id, metadata, vector)
                print("\nFetch-back verification passed:")
                print("  Chunk ID, original text and every metadata field match.")
                print(f"  All {EMBEDDING_DIMENSIONS} vector values match within numeric tolerance.")
                print(f"  Record is stored in namespace {NAMESPACE}.")
                print("No local files changed. This verifies storage, not retrieval quality.")
                return
            if attempt < 5:
                time.sleep(2)
        raise ValueError("Record is not visible yet. Upsert was acknowledged; rerun later to verify/reuse it.")
    except ValueError as error:
        parser.exit(1, f"Validation stopped while {stage}: {error}\n")
    except Exception as error:
        # Never print credential-bearing request payloads or raw exception text.
        parser.exit(1, f"Operation failed while {stage} ({type(error).__name__}).\nAn acknowledged or interrupted upload may already exist; rerunning checks the same ID first.\n")
    finally:
        del api_key


if __name__ == "__main__":
    main()
