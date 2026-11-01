"""Lesson 4: inspect embedding inputs without calling an embedding service.

Inventory (standard library only):
    uv run python check_chunk_lengths.py --inventory

Preliminary token counts (requires tokenizers and a local tokenizer.json):
    uv run python check_chunk_lengths.py --tokenizer work/tokenizer/tokenizer.json --tokenizer-label "Llama-compatible tokenizer; hosted match unverified"

Neither mode modifies the corpus or export, downloads files, or calls Pinecone.
Local counts do not certify Pinecone's hosted input limit or preprocessing.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


PROJECT_FOLDER = Path(__file__).resolve().parent
DEFAULT_EXPORT = PROJECT_FOLDER / "work" / "chunks.jsonl"
PUBLISHED_INPUT_LIMIT = 2048
REVIEW_THRESHOLD = 1800  # Our review margin, not a provider-defined limit.


def embedding_text(record: dict) -> str:
    """Proposed embedding input: concise source labels, context, then evidence.

    Future embedding code should reuse this function if we adopt this format.
    Metadata remains separate in the saved chunk; nothing is rewritten here.
    """
    metadata = record["metadata"]
    labels = []
    for key in (
        "source", "document_type", "hostname", "scenario_id",
        "snapshot_at", "snapshot_is_simulated", "evidence_is_simulated",
        "event_times_are_assigned", "scenario_is_independent", "baseline_basis",
    ):
        if key in metadata:
            labels.append(f"{key}: {json.dumps(metadata[key], ensure_ascii=False)}")

    parts = ["\n".join(labels)]
    context = metadata.get("document_context", "")
    if context and context != record["page_content"]:
        parts.append("Document context:\n" + context)
    parts.append("Evidence:\n" + record["page_content"])
    return "\n\n".join(parts)


def read_chunks(path: Path) -> list[dict]:
    """Read JSONL and reject empty text, missing provenance and duplicate IDs."""
    records = []
    seen_ids = set()
    with path.open(encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(f"Unexpected blank JSONL record at line {line_number}.")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Record {line_number} must be a JSON object.")
            chunk_id = record.get("id")
            content = record.get("page_content")
            metadata = record.get("metadata")
            if not isinstance(chunk_id, str) or not chunk_id or chunk_id in seen_ids:
                raise ValueError(f"Missing or duplicate chunk ID at record {line_number}.")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"Missing chunk text at record {line_number}.")
            if not isinstance(metadata, dict):
                raise ValueError(f"Missing metadata at record {line_number}.")
            source = metadata.get("source")
            if not isinstance(source, str):
                raise ValueError(f"Missing source at record {line_number}.")
            source_path = PurePosixPath(source)
            if (
                source_path.is_absolute() or not source_path.parts
                or source_path.parts[0] != "corpus" or ".." in source_path.parts
                or "\\" in source
            ):
                raise ValueError(f"Expected a relative corpus source at record {line_number}.")
            if not isinstance(metadata.get("document_type"), str):
                raise ValueError(f"Missing document type at record {line_number}.")
            if not isinstance(metadata.get("document_context", ""), str):
                raise ValueError(f"Document context must be text at record {line_number}.")
            seen_ids.add(chunk_id)
            records.append(record)
    if not records:
        raise ValueError("The chunk export is empty.")
    return records


def description(record: dict) -> str:
    metadata = record["metadata"]
    return (
        f"{metadata['source']} | {metadata.get('section', '')} | "
        f"lines {metadata.get('line_start', '?')}-{metadata.get('line_end', '?')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_EXPORT, help="JSONL export to read.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inventory", action="store_true", help="Rank by characters; no tokenizer needed.")
    mode.add_argument("--tokenizer", type=Path, help="Existing local tokenizer.json; never downloaded automatically.")
    parser.add_argument("--tokenizer-label", help="Tokenizer source/revision, recorded in terminal output.")
    parser.add_argument("--top", type=int, default=15, help="Number of largest inputs to print.")
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be at least 1.")
    if args.tokenizer is not None and not args.tokenizer_label:
        parser.error("Supply --tokenizer-label to identify the tokenizer source/revision.")
    if args.tokenizer_label and args.tokenizer is None:
        parser.error("--tokenizer-label requires --tokenizer.")

    try:
        records = read_chunks(args.input)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Cannot read chunk export: {error}\n")

    print(f"Read {len(records)} chunks from {args.input}.")
    print("IDs and basic source records checked; export is unchanged.")
    print("Prepared text includes proposed source labels and available document context.")

    if args.inventory:
        ranked = sorted(records, key=lambda record: len(embedding_text(record)), reverse=True)
        print("\nCharacter inventory only: these are NOT token counts or fit decisions.")
        for record in ranked[:args.top]:
            print(f"  {len(embedding_text(record)):6} prepared characters | {description(record)}")
        print("\nNext: use a reviewed local tokenizer for preliminary token counts.")
    else:
        if not args.tokenizer.is_file():
            parser.error("Tokenizer file does not exist. No download was attempted.")
        try:
            from tokenizers import Tokenizer
        except ImportError:
            parser.exit(1, "The tokenizers package is not installed. No installation was attempted.\n")

        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        # A tokenizer file may include saved padding or truncation settings.
        # Disable both so the count represents the complete input.
        tokenizer.no_truncation()
        tokenizer.no_padding()
        print(f"\nTokenizer label: {args.tokenizer_label}")
        print(f"Tokenizer SHA256: {hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()}")
        print("PRELIMINARY: hosted tokenizer/preprocessing equivalence is not verified.")
        print("Counts use this tokenizer's configured special-token processing.")
        print(f"Published input limit: {PUBLISHED_INPUT_LIMIT}; our review threshold: {REVIEW_THRESHOLD}.")
        rows = []
        for record in records:
            raw_count = len(tokenizer.encode(record["page_content"], add_special_tokens=True).ids)
            prepared_count = len(tokenizer.encode(embedding_text(record), add_special_tokens=True).ids)
            rows.append((prepared_count, raw_count, record))
        rows.sort(key=lambda row: row[0], reverse=True)
        for prepared_count, raw_count, record in rows[:args.top]:
            print(f"  prepared={prepared_count:5} raw={raw_count:5} | {description(record)}")

        flagged = [row for row in rows if row[0] >= REVIEW_THRESHOLD]
        oversized = [row for row in rows if row[0] > PUBLISHED_INPUT_LIMIT]
        print(f"\nInputs at/above our review threshold: {len(flagged)}")
        print(f"Inputs above the published limit under this local count: {len(oversized)}")
        for prepared_count, _, record in flagged:
            print(f"  REVIEW {prepared_count} | {description(record)} | id={record['id']}")
        print("Local counts do not certify hosted acceptance, even below the threshold.")
        print("Later embedding requests must reject truncation with truncate=NONE.")

    print("\nNo files changed, downloads made, or embedding/Pinecone calls made.")


if __name__ == "__main__":
    main()
