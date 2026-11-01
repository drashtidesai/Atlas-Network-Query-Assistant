"""Lesson 9: inspect dense retrieval before adding BM25 or answer generation.

Preview: uv run python retrieve_dense.py
Search: uv run python retrieve_dense.py --execute
Custom question: uv run python retrieve_dense.py --question "How does VRRP tracking work?" --execute

Execution embeds one question via LangChain and queries Pinecone read-only.
No answer-generation model is called. No records or local files are written.
"""

import argparse
import getpass
import math
import warnings
from pathlib import Path

from check_chunk_lengths import DEFAULT_EXPORT, embedding_text, read_chunks
from embed_one_chunk import validate_vector
from rag_settings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, INDEX_NAME, NAMESPACE
from setup_pinecone_index import check_configuration
from upload_one_chunk import storage_metadata


DEFAULT_QUESTION = (
    "In synthetic FAIL-01, why did DXB-R01 remain the VRRP master "
    "after WAN-A silently failed?"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_EXPORT, help="Local export used to verify returned evidence.")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Question to embed and search with.")
    parser.add_argument("--top-k", type=int, default=5, help="Maximum matches, 1 to 20.")
    parser.add_argument("--show-text", action="store_true", help="Print complete evidence instead of 1,000-character excerpts.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preview", action="store_true", help="Local plan only (default).")
    modes.add_argument("--execute", action="store_true", help="Embed the question and run a read-only vector search.")
    args = parser.parse_args()
    question = args.question.strip()
    if not question:
        parser.error("--question cannot be empty.")
    if not 1 <= args.top_k <= 20:
        parser.error("--top-k must be between 1 and 20.")

    try:
        records = read_chunks(args.input)
        expected_metadata = {
            record["id"]: storage_metadata(record, embedding_text(record))
            for record in records
        }
    except (OSError, ValueError) as error:
        parser.exit(1, f"Cannot read/validate the local export: {error}\n")

    print(f"Mode: {'EXECUTE' if args.execute else 'PREVIEW'}")
    print(f"Question: {question}")
    print(f"Index: {INDEX_NAME} | namespace: {NAMESPACE}")
    print(f"Model: {EMBEDDING_MODEL} | dimensions: {EMBEDDING_DIMENSIONS}")
    print("Question input type: query | truncation: NONE")
    print(f"Top-k: {args.top_k} | local reference chunks: {len(records)}")
    print("Retrieval: dense only; no BM25, reranking or generated answer yet.")
    print("Search includes all evidence types; FAIL-01 in the question is not a metadata filter.")
    print("Model/input-format metadata filters restrict results to our embedding configuration.")
    print("Returned IDs, text, metadata and input fingerprints must match the local export.")
    print("Cosine scores rank similarity; they are not confidence percentages or proof.")

    if not args.execute:
        print("\nPreview complete: no key requested, API calls made or files changed.")
        return

    try:
        from langchain_pinecone import PineconeEmbeddings
        from pinecone import Pinecone
    except ImportError:
        parser.exit(1, "Missing LangChain/Pinecone dependencies. No API call was made.\n")

    print("\nExecution sends the question for embedding and searches stored vectors; usage may be billed.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            api_key = getpass.getpass("Enter your existing Pinecone API key (hidden): ").strip()
    except (getpass.GetPassWarning, EOFError, KeyboardInterrupt):
        parser.exit(1, "\nKey entry cancelled or hidden input unavailable.\n")
    if not api_key:
        parser.exit(1, "No key entered. No API call was made.\n")

    stage = "checking index"
    try:
        client = Pinecone(api_key=api_key)
        description = client.describe_index(INDEX_NAME).to_dict()
        check_configuration(description)
        if description.get("status", {}).get("ready") is not True:
            raise ValueError("Index is not ready.")
        index = client.Index(host=description["host"])

        stage = "embedding question"
        embeddings = PineconeEmbeddings(
            model=EMBEDDING_MODEL,
            dimension=EMBEDDING_DIMENSIONS,
            pinecone_api_key=api_key,
            document_params={"input_type": "passage", "dimension": EMBEDDING_DIMENSIONS, "truncate": "NONE"},
            query_params={"input_type": "query", "dimension": EMBEDDING_DIMENSIONS, "truncate": "NONE"},
        )
        vector = validate_vector([embeddings.embed_query(question)])

        stage = "searching Pinecone"
        response = index.query(
            vector=vector,
            top_k=args.top_k,
            namespace=NAMESPACE,
            include_metadata=True,
            include_values=False,
            filter={
                "embedding_model": {"$eq": EMBEDDING_MODEL},
                "embedding_dimension": {"$eq": EMBEDDING_DIMENSIONS},
                "embedding_input_format": {"$eq": "source-labels-context-evidence-v1"},
            },
        )

        stage = "verifying retrieved evidence"
        matches = response.matches or []
        seen_ids = set()
        for match in matches:
            if match.id in seen_ids or match.id not in expected_metadata:
                raise ValueError("Search returned a duplicate ID or an ID outside the local export.")
            if match.metadata != expected_metadata[match.id]:
                raise ValueError("Retrieved metadata/text differs from the local export. Review indexing state.")
            if match.score is None or not math.isfinite(match.score):
                raise ValueError("Search returned an invalid similarity score.")
            seen_ids.add(match.id)

        print(f"\nRetrieved {len(matches)} verified matches:")
        for rank, match in enumerate(matches, start=1):
            metadata = match.metadata
            ranges = metadata.get("source_line_ranges") or [
                f"{metadata['line_start']}-{metadata['line_end']}"
            ]
            print(f"\n{rank}. Cosine score: {match.score:.4f}")
            print(f"   Source: {metadata['source']} | exact lines: {', '.join(ranges)}")
            print(f"   Type: {metadata['document_type']} | section: {metadata.get('section', '')}")
            print(f"   Chunk ID: {match.id}")
            if "scenario_id" in metadata:
                start = metadata.get("chunk_event_time_start", metadata.get("event_time_start"))
                end = metadata.get("chunk_event_time_end", metadata.get("event_time_end"))
                print(f"   Synthetic scenario: {metadata['scenario_id']} | excerpt UTC range: {start} -> {end}")
            if "snapshot_at" in metadata:
                print(f"   Snapshot: {metadata['snapshot_at']} | simulated: {metadata.get('snapshot_is_simulated')}")
            if metadata.get("document_context"):
                print(f"   Document context: {metadata['document_context']}")
            evidence = metadata["text"]
            print("   Evidence:" if args.show_text else "   Evidence excerpt (up to 1,000 characters):")
            print(evidence if args.show_text else evidence[:1000])
            if not args.show_text and len(evidence) > 1000:
                print("   [Display shortened; use --show-text for the complete evidence.]")

        if not matches:
            print("No matching records found. Check namespace/indexing state before drawing conclusions.")
        print("\nNo answer generated, records changed or local files written.")
        print("Review whether the evidence covers the question before adding hybrid retrieval.")
    except KeyboardInterrupt:
        parser.exit(1, f"\nInterrupted while {stage}. No records were written.\n")
    except ValueError as error:
        parser.exit(1, f"Validation stopped while {stage}: {error}\n")
    except Exception as error:
        parser.exit(1, f"Operation failed while {stage} ({type(error).__name__}). No records were written.\n")
    finally:
        del api_key


if __name__ == "__main__":
    main()
