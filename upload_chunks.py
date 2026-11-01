"""Lesson 8: preview or upload all exported chunks in resumable batches.

Preview: uv run python upload_chunks.py
Execute: uv run python upload_chunks.py --execute

Prompts for an existing key only in execution mode. Checks all expected remote
IDs before writing; matching records are reused and conflicts stop the run.
No local files, indexes or unrelated records are changed. Embedding and database
usage may be billed. An interrupted run can be restarted with the same command.
"""

import argparse
import getpass
import math
import time
import warnings
from collections import Counter
from pathlib import Path

from check_chunk_lengths import DEFAULT_EXPORT, embedding_text, read_chunks
from embed_one_chunk import validate_vector
from rag_settings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, INDEX_NAME, NAMESPACE
from setup_pinecone_index import check_configuration
from upload_one_chunk import storage_metadata, verify_record


def batches(items: list, size: int):
    """Yield successive small lists, including the shorter final batch."""
    for start in range(0, len(items), size):
        yield items[start:start + size]


def classify_existing(index, items: list[dict]) -> tuple[list[dict], int]:
    """Fetch expected IDs; verify matching records before marking them reusable."""
    fetched = index.fetch(ids=[item["id"] for item in items], namespace=NAMESPACE).vectors
    missing = []
    reused = 0
    for item in items:
        saved = fetched.get(item["id"])
        if saved is None:
            missing.append(item)
        else:
            verify_record(saved.to_dict(), item["id"], item["metadata"])
            reused += 1
    return missing, reused


def verify_uploaded_batch(index, items: list[dict], vectors: list[list[float]]) -> None:
    """Allow brief propagation delay; retry reads only, not embedding or upload."""
    pending = {item["id"]: (item, vector) for item, vector in zip(items, vectors)}
    for attempt in range(6):
        fetched = index.fetch(ids=list(pending), namespace=NAMESPACE).vectors
        for chunk_id in list(pending):
            if chunk_id in fetched:
                item, vector = pending[chunk_id]
                verify_record(fetched[chunk_id].to_dict(), chunk_id, item["metadata"], vector)
                del pending[chunk_id]
        if not pending:
            return
        if attempt < 5:
            time.sleep(2)
    raise ValueError(f"{len(pending)} acknowledged records are not visible yet. Rerun later to verify/reuse them.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding/upload batch size, 1 to 32.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true", help="Local plan only (default).")
    mode.add_argument("--execute", action="store_true", help="Check remote records, embed missing chunks and upload.")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 32:
        parser.error("--batch-size must be between 1 and 32.")

    try:
        records = read_chunks(args.input)
        items = []
        for record in records:
            text = embedding_text(record)
            items.append({"id": record["id"], "text": text, "metadata": storage_metadata(record, text)})
    except (OSError, ValueError) as error:
        parser.exit(1, f"Cannot prepare export: {error}\n")

    print(f"Mode: {'EXECUTE' if args.execute else 'PREVIEW'}")
    print(f"Export: {args.input}")
    print(f"Expected chunks: {len(items)} | source files: {len({r['metadata']['source'] for r in records})}")
    for kind, count in sorted(Counter(r["metadata"]["document_type"] for r in records).items()):
        print(f"  {kind}: {count}")
    print(f"Index: {INDEX_NAME} | namespace: {NAMESPACE}")
    print(f"Model: {EMBEDDING_MODEL} | dimensions: {EMBEDDING_DIMENSIONS}")
    print("Input type: passage | truncation: NONE")
    print(f"Batch size: {args.batch_size}; at most {math.ceil(len(items) / args.batch_size)} upload batches if all records are missing.")
    print("All local IDs and metadata payloads checked.")
    print("Remote preflight: verify all expected IDs; reuse matching records, stop on conflicts.")
    print("Every new batch: embed, validate vectors, upsert, then fetch back and compare.")
    print("Final verification checks every expected ID, text, metadata and vector shape.")
    print("Unrelated IDs are not removed; namespace contents are not assumed to equal this export.")

    if not args.execute:
        print("\nPreview complete: remote record counts are unknown; no key requested or API calls made.")
        return

    try:
        from langchain_pinecone import PineconeEmbeddings
        from pinecone import Pinecone
    except ImportError:
        parser.exit(1, "Missing LangChain/Pinecone dependencies. No API calls were made.\n")

    print("\nExecution sends missing chunk inputs to Pinecone and stores vectors, text and metadata.")
    print("Embedding/database usage may be billed; the key is not saved by this script.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            api_key = getpass.getpass("Enter your existing Pinecone API key (hidden): ").strip()
    except (getpass.GetPassWarning, EOFError, KeyboardInterrupt):
        parser.exit(1, "\nKey entry cancelled or hidden input unavailable.\n")
    if not api_key:
        parser.exit(1, "No key entered. No API calls were made.\n")

    stage = "checking index"
    uploaded = 0
    reused = 0
    try:
        client = Pinecone(api_key=api_key)
        description = client.describe_index(INDEX_NAME).to_dict()
        check_configuration(description)
        if description.get("status", {}).get("ready") is not True:
            raise ValueError("Index is not ready.")
        index = client.Index(host=description["host"])

        stage = "remote preflight (no writes yet)"
        missing = []
        for group in batches(items, 100):
            absent, matching = classify_existing(index, group)
            missing.extend(absent)
            reused += matching
        print(f"\nPreflight passed: {reused} matching records; {len(missing)} missing.")

        embeddings = None
        if missing:
            stage = "initializing embedding model"
            embeddings = PineconeEmbeddings(
                model=EMBEDDING_MODEL,
                dimension=EMBEDDING_DIMENSIONS,
                pinecone_api_key=api_key,
                document_params={"input_type": "passage", "dimension": EMBEDDING_DIMENSIONS, "truncate": "NONE"},
                query_params={"input_type": "query", "dimension": EMBEDDING_DIMENSIONS, "truncate": "NONE"},
            )

        batch_total = math.ceil(len(missing) / args.batch_size)
        for batch_number, group in enumerate(batches(missing, args.batch_size), start=1):
            stage = f"rechecking batch {batch_number}/{batch_total}"
            # Check again immediately before writing, including on resumed runs.
            group, matching = classify_existing(index, group)
            reused += matching
            if not group:
                continue
            stage = f"embedding batch {batch_number}/{batch_total}"
            print(f"{stage}: {len(group)} inputs", flush=True)
            vectors = embeddings.embed_documents([item["text"] for item in group])
            if len(vectors) != len(group):
                raise ValueError("Embedding response count differs from input count.")
            for vector in vectors:
                validate_vector([vector])

            stage = f"uploading batch {batch_number}/{batch_total}"
            response = index.upsert(
                vectors=[
                    {"id": item["id"], "values": vector, "metadata": item["metadata"]}
                    for item, vector in zip(group, vectors)
                ],
                namespace=NAMESPACE,
            )
            if response.upserted_count != len(group):
                raise ValueError("Upsert acknowledgement count differs from the submitted batch.")
            stage = f"verifying batch {batch_number}/{batch_total}"
            verify_uploaded_batch(index, group, vectors)
            uploaded += len(group)
            print(f"  Verified batch; newly uploaded so far: {uploaded}", flush=True)

        stage = "final expected-record verification"
        verified = 0
        for group in batches(items, 100):
            absent, matching = classify_existing(index, group)
            if absent:
                raise ValueError(f"{len(absent)} expected records are not visible in the final check; rerun later.")
            verified += matching
        print("\nBatch upload checkpoint passed:")
        print(f"  Expected records verified: {verified}/{len(items)}")
        print(f"  Existing matching records reused: {reused}")
        print(f"  New records uploaded and fetched back: {uploaded}")
        print("  Every expected record has matching text, metadata and a valid 1,024-value vector.")
        print("  Newly uploaded vectors also matched their generated values within tolerance.")
        print("No local files changed or unrelated records deleted. Retrieval quality is not tested yet.")
    except KeyboardInterrupt:
        parser.exit(1, f"\nInterrupted while {stage}. Verified new uploads: {uploaded}. Rerun to check/reuse stored records.\n")
    except ValueError as error:
        parser.exit(1, f"Validation stopped while {stage}: {error}\nVerified new uploads: {uploaded}. Earlier or interrupted writes may remain; rerun checks IDs first.\n")
    except Exception as error:
        parser.exit(1, f"Operation failed while {stage} ({type(error).__name__}).\nVerified new uploads: {uploaded}. Earlier or interrupted writes may remain. Rerun checks IDs before embedding again.\n")
    finally:
        del api_key


if __name__ == "__main__":
    main()
