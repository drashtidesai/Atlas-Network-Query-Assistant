"""Lesson 6: preview, inspect or create the project's dense-vector index.

Preview (default; no API calls): uv run python setup_pinecone_index.py
Read-only remote check: uv run python setup_pinecone_index.py --check
Create if absent: uv run python setup_pinecone_index.py --execute

Remote modes prompt for an existing key. No embeddings, upserts, deletions,
namespace creation, local file writes or BM25 setup happen in this script.
"""

import argparse
import getpass
import warnings

from rag_settings import (
    CLOUD, EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, INDEX_NAME,
    NAMESPACE, REGION, SIMILARITY_METRIC,
)


def check_configuration(index: dict) -> None:
    """Refuse to use a same-name index with different structural settings."""
    serverless = index.get("spec", {}).get("serverless", {})
    expected = {
        "name": INDEX_NAME,
        "dimension": EMBEDDING_DIMENSIONS,
        "metric": SIMILARITY_METRIC,
        "vector_type": "dense",
    }
    for key, value in expected.items():
        if index.get(key) != value:
            raise ValueError(f"Existing index has a different {key}; it was not modified.")
    if serverless.get("cloud") != CLOUD or serverless.get("region") != REGION:
        raise ValueError("Existing index has a different cloud/region; it was not modified.")
    if index.get("embed"):
        raise ValueError("Expected a bring-your-own-vectors index, not integrated embedding.")
    model_tag = index.get("tags", {}).get("embedding_model")
    if model_tag is not None and model_tag != EMBEDDING_MODEL:
        raise ValueError("Existing embedding-model tag differs; index was not modified.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true", help="Show the plan only (default).")
    mode.add_argument("--check", action="store_true", help="Inspect an existing remote index without creating one.")
    mode.add_argument("--execute", action="store_true", help="Create the remote index if absent; may consume account resources.")
    args = parser.parse_args()

    print(f"Mode: {'EXECUTE' if args.execute else 'CHECK' if args.check else 'PREVIEW'}")
    print(f"Index name: {INDEX_NAME}")
    print("Index type: serverless dense vectors; embeddings generated separately via LangChain")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Dimensions: {EMBEDDING_DIMENSIONS}")
    print(f"Similarity metric: {SIMILARITY_METRIC}")
    print(f"Cloud / region: {CLOUD} / {REGION}")
    print(f"Planned upload namespace: {NAMESPACE} (not created by this script)")
    print("Planned BM25: local LangChain retriever over the same chunks; not implemented yet.")

    if not (args.check or args.execute):
        print("\nPreview complete: no API key requested, network calls made or files changed.")
        return

    try:
        from pinecone import Pinecone, ServerlessSpec
    except ImportError:
        parser.exit(1, "Pinecone SDK is unavailable. Check the installed project dependencies.\n")

    if args.execute:
        print("\nExecution may create an index in your Pinecone account, subject to its plan and quotas.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            api_key = getpass.getpass("Enter your existing Pinecone API key (hidden): ").strip()
    except (getpass.GetPassWarning, EOFError, KeyboardInterrupt):
        parser.exit(1, "\nKey entry cancelled or hidden input unavailable. Use an interactive terminal.\n")
    if not api_key:
        parser.exit(1, "No key entered. No API call was made.\n")

    try:
        client = Pinecone(api_key=api_key)
        exists = client.has_index(INDEX_NAME)
        if not exists and args.check:
            print("\nIndex does not exist. Read-only check created nothing.")
            return
        if not exists:
            print("\nSubmitting index creation request...")
            client.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIMENSIONS,
                metric=SIMILARITY_METRIC,
                vector_type="dense",
                spec=ServerlessSpec(cloud=CLOUD, region=REGION),
                tags={"project": "atlas-network-rag", "embedding_model": EMBEDDING_MODEL},
                # Return after acceptance rather than waiting indefinitely.
                timeout=-1,
            )
            print("Creation request accepted. Checking current status...")
        else:
            print("\nIndex already exists; checking it without changing its settings.")

        index = client.describe_index(INDEX_NAME).to_dict()
        check_configuration(index)
        print("Structural settings match the project plan.")
        print(f"State: {index.get('status', {}).get('state', 'unknown')}")
        ready = index.get("status", {}).get("ready") is True
        print(f"Ready: {ready}")
        if not ready:
            print("Provisioning may still be in progress. Run this script with --check later.")
        print("Index structure does not verify the embedding model of any existing records.")
        print("No embeddings generated, records uploaded, namespaces created or local files written.")
    except ValueError as error:
        parser.exit(1, f"Index configuration check failed: {error}\n")
    except Exception as error:
        # Do not print request payloads or exception details containing credentials.
        parser.exit(1, f"Pinecone operation failed ({type(error).__name__}).\nCheck account permissions, quota, region availability and connectivity.\nIf creation was submitted, use --check to inspect the result before retrying.\n")
    finally:
        del api_key


if __name__ == "__main__":
    main()
