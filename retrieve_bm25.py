"""Lesson 10: inspect local BM25 results before combining retrieval rankings.

Preview: uv run python retrieve_bm25.py
Local search: uv run python retrieve_bm25.py --execute

Execution requires langchain-community and rank-bm25. No API keys, embedding
requests, Pinecone calls, downloads or file writes are made by this script.
"""

import argparse
import re
from pathlib import Path

from check_chunk_lengths import DEFAULT_EXPORT, embedding_text, read_chunks
from retrieve_dense import DEFAULT_QUESTION


# Keep negations and networking words; omit only these common question words.
STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "of", "to", "for", "and", "or",
    "is", "are", "was", "were", "be", "did", "does", "do", "why", "how",
    "what", "which", "with", "after", "before", "it", "its", "that", "this",
}


def network_tokens(text: str) -> list[str]:
    """Lowercase words while retaining identifiers, addresses and interfaces.

    Examples: FAIL-01 -> fail-01; 10.10.99.1 -> 10.10.99.1.
    Also expose underscore-separated fields: VRRP_110 -> vrrp_110, vrrp, 110.
    This is lexical processing, not the embedding model's tokenizer.
    """
    tokens = []
    for token in re.findall(r"[a-z0-9]+(?:[._:/-][a-z0-9]+)*", text.lower()):
        if token not in STOP_WORDS:
            tokens.append(token)
        if "_" in token:
            tokens.extend(part for part in token.split("_") if part not in STOP_WORDS)
    return tokens


def build_bm25(records: list[dict], top_k: int):
    """Build LangChain's BM25 retriever in memory from prepared chunk texts."""
    from langchain_core.documents import Document
    from langchain_community.retrievers import BM25Retriever

    documents = [
        Document(
            id=record["id"],
            page_content=embedding_text(record),
            metadata=record["metadata"].copy(),
        )
        for record in records
    ]
    return BM25Retriever.from_documents(
        documents,
        preprocess_func=network_tokens,
        k=min(top_k, len(documents)),
    )


def bm25_results(retriever, records: list[dict], question: str) -> list[dict]:
    """Return ranked IDs/scores with the original source evidence for display."""
    tokens = network_tokens(question)
    scores = retriever.vectorizer.get_scores(tokens)
    positions = {record["id"]: index for index, record in enumerate(records)}
    results = []
    for document in retriever.invoke(question):
        if document.id not in positions:
            raise ValueError("BM25 returned an ID outside the local export.")
        position = positions[document.id]
        record = records[position]
        if document.page_content != embedding_text(record) or document.metadata != record["metadata"]:
            raise ValueError("Retrieved BM25 document differs from the source export.")
        matched_terms = sorted(set(tokens) & set(network_tokens(document.page_content)))
        # BM25 can return arbitrary zero-overlap records to fill top-k slots.
        if not matched_terms:
            continue
        results.append({
            "id": document.id,
            "score": float(scores[position]),
            "record": record,
            "matched_terms": matched_terms,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--top-k", type=int, default=5, help="Maximum matches, 1 to 20.")
    parser.add_argument("--show-text", action="store_true", help="Show complete evidence instead of excerpts.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preview", action="store_true", help="Local plan only (default).")
    modes.add_argument("--execute", action="store_true", help="Build BM25 in memory and search locally.")
    args = parser.parse_args()
    question = args.question.strip()
    if not question or not network_tokens(question):
        parser.error("Question must contain searchable terms after preprocessing.")
    if not 1 <= args.top_k <= 20:
        parser.error("--top-k must be between 1 and 20.")
    try:
        records = read_chunks(args.input)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Cannot read chunk export: {error}\n")

    print(f"Mode: {'EXECUTE (local only)' if args.execute else 'PREVIEW'}")
    print(f"Question: {question}")
    print(f"Chunks: {len(records)} | top-k: {args.top_k}")
    print(f"BM25 query tokens: {network_tokens(question)}")
    print("Indexed text: same prepared source labels/context/evidence used for dense embeddings.")
    print("Identifiers retain hyphens, dots, colons and slashes; no stemming is applied.")
    print("No mandatory scenario filter: keyword matches affect rank, not eligibility.")
    print("BM25 scores are lexical rankings, not cosine scores or confidence percentages.")
    print("Results without any matching query tokens are omitted.")
    if not args.execute:
        print("\nPreview complete: no BM25 index built, API calls made or files changed.")
        return

    try:
        retriever = build_bm25(records, args.top_k)
        results = bm25_results(retriever, records, question)
    except ImportError:
        parser.exit(1, "Local BM25 requires two packages. Install with:\nuv add langchain-community rank-bm25\nNo API calls were made.\n")
    except ValueError as error:
        parser.exit(1, f"Local retrieval validation failed: {error}\n")

    print(f"\nRetrieved {len(results)} verified local BM25 matches:")
    for rank, result in enumerate(results, start=1):
        record = result["record"]
        metadata = record["metadata"]
        ranges = metadata.get("source_line_ranges") or [f"{metadata['line_start']}-{metadata['line_end']}"]
        print(f"\n{rank}. BM25 score: {result['score']:.4f}")
        print(f"   Source: {metadata['source']} | exact lines: {', '.join(ranges)}")
        print(f"   Type: {metadata['document_type']} | section: {metadata.get('section', '')}")
        print(f"   Chunk ID: {record['id']}")
        print(f"   Matching query tokens: {result['matched_terms']}")
        if metadata.get("scenario_id"):
            print(f"   Synthetic scenario: {metadata['scenario_id']}")
        if metadata.get("document_context"):
            print(f"   Document context: {metadata['document_context']}")
        evidence = record["page_content"]
        print("   Evidence:" if args.show_text else "   Evidence excerpt (up to 1,000 characters):")
        print(evidence if args.show_text else evidence[:1000])
        if not args.show_text and len(evidence) > 1000:
            print("   [Display shortened; use --show-text for complete evidence.]")
    if not results:
        print("No candidate had matching query tokens; this does not establish missing corpus evidence.")
    print("\nNo API key needed, network calls made, answers generated or files changed.")
    print("The local BM25 index exists only in memory for this run.")


if __name__ == "__main__":
    main()
