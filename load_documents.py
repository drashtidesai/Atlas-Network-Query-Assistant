"""Lesson 2: turn corpus files into LangChain Documents.

Run: uv run python load_documents.py
Optional local export: uv run python load_documents.py --export
No embeddings, AI requests, or Pinecone calls happen here.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document


PROJECT_FOLDER = Path(__file__).resolve().parent
CORPUS_FOLDER = PROJECT_FOLDER / "corpus"

# These labels distinguish design intent, configuration snapshots and synthetic logs.
DOCUMENT_TYPES = {
    "intended-configs": "intended_config",
    "running-config-snapshots": "running_config",
    "design": "design_document",
    "topology": "topology_document",
    "references": "technology_reference",
    "synthetic-logs": "synthetic_log",
}


def synthetic_log_metadata(content: str, filename: str) -> dict:
    """Validate the teaching-log format and describe its synthetic timeline."""
    lines = content.splitlines()
    header = re.fullmatch(
        r"SYNTHETIC=true \| scenario_id=(FAIL-\d{2}) \| (.+)", lines[0]
    )
    if not header:
        raise ValueError(f"Synthetic log needs a scenario and simulation label: {filename}")
    scenario_id, scenario_name = header.groups()
    if not filename.startswith(f"test-{scenario_id[-2:]}-"):
        raise ValueError(f"Scenario ID does not match filename: {filename}")
    if len(lines) < 4 or lines[1:3] != [
        "Assigned simulated event times on 2026-10-24 UTC; not hardware observations.",
        "Independent scenario using intended baseline; normalized teaching records.",
    ]:
        raise ValueError(f"Expected the reviewed teaching-log context: {filename}")

    timestamps = []
    observers = set()
    previous_time = None
    for line in lines[3:]:
        fields = line.split(" | ", 3)
        if len(fields) != 4 or not all(fields):
            raise ValueError(f"Malformed synthetic log record: {filename}")
        timestamp, observer, record_kind, details = fields
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", timestamp):
            raise ValueError(f"Expected a millisecond UTC timestamp: {filename}")
        event_time = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
        if event_time.date().isoformat() != "2026-10-24":
            raise ValueError(f"Event date differs from the reviewed log context: {filename}")
        if previous_time is not None and event_time < previous_time:
            raise ValueError(f"Synthetic log timestamps are out of order: {filename}")
        if record_kind not in {"STATE", "EVENT", "PROBE", "INJECTION", "RESTORE", "COLLECTOR", "SERVER"}:
            raise ValueError(f"Unknown synthetic record kind: {record_kind}")
        timestamps.append(timestamp)
        observers.add(observer)
        previous_time = event_time

    return {
        "scenario_id": scenario_id,
        "scenario_name": scenario_name,
        "evidence_is_simulated": True,
        "event_times_are_assigned": True,
        "scenario_is_independent": True,
        "baseline_basis": "intended_config",
        "event_time_start": timestamps[0],
        "event_time_end": timestamps[-1],
        "event_count": len(timestamps),
        "observers": sorted(observers),
        "log_format": "normalized_teaching_records",
    }


def load_documents() -> list[Document]:
    """Read only corpus/, returning one Document per file for now."""
    if not CORPUS_FOLDER.is_dir():
        raise FileNotFoundError(f"Corpus folder does not exist: {CORPUS_FOLDER}")

    documents = []

    for path in sorted(CORPUS_FOLDER.rglob("*")):
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(CORPUS_FOLDER.resolve()):
            raise ValueError(f"Source points outside the corpus: {path}")
        if path.suffix.lower() not in {".txt", ".csv"}:
            raise ValueError(f"Unsupported corpus file: {path.name}")

        content = path.read_text(encoding="utf-8-sig").strip()
        if not content:
            raise ValueError(f"Corpus file is empty: {path.name}")

        folder_name = path.relative_to(CORPUS_FOLDER).parts[0]
        if folder_name not in DOCUMENT_TYPES:
            raise ValueError(f"Unknown source folder: {folder_name}")

        metadata = {
            "source": path.relative_to(PROJECT_FOLDER).as_posix(),
            "filename": path.name,
            "document_type": DOCUMENT_TYPES[folder_name],
            "file_format": path.suffix.lstrip("."),
            "line_start": 1,
            "line_end": len(content.splitlines()),
        }

        # Extract a device name only from device configuration filenames.
        if metadata["document_type"] in {"intended_config", "running_config"}:
            hostname = path.stem.split("_")[0]
            metadata["hostname"] = hostname
            metadata["site"] = hostname.split("-")[0]

        if metadata["document_type"] == "running_config":
            timestamp = re.search(r"(\d{4}-\d{2}-\d{2})T(\d{2})(\d{2})(\d{2})Z", path.stem)
            if not timestamp:
                raise ValueError(f"Running snapshot needs a timestamp: {path.name}")
            date, hour, minute, second = timestamp.groups()
            metadata["snapshot_at"] = f"{date}T{hour}:{minute}:{second}Z"
            metadata["snapshot_is_simulated"] = True

        if metadata["document_type"] == "synthetic_log":
            metadata.update(synthetic_log_metadata(content, path.name))

        # This fingerprint changes if the source path or its contents change.
        fingerprint = hashlib.sha256(
            (metadata["source"] + "\n" + content).encode("utf-8")
        ).hexdigest()

        document = Document(
            id=fingerprint,
            page_content=content,
            metadata=metadata,
        )
        documents.append(document)

    if not documents:
        raise ValueError("No supported documents were found in corpus/.")
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true", help="Save a local JSONL preview under work/.")
    args = parser.parse_args()

    documents = load_documents()
    print(f"Loaded {len(documents)} LangChain Documents from corpus/.\n")

    counts = Counter(doc.metadata["document_type"] for doc in documents)
    for document_type, count in sorted(counts.items()):
        print(f"  {document_type:24} {count}")

    print("\nExample metadata for DXB-R01:")
    for doc in documents:
        if doc.metadata.get("hostname") == "DXB-R01":
            print(json.dumps(doc.metadata, indent=2))

    print("\nExample metadata for the first synthetic log:")
    log_example = next(
        (doc for doc in documents if doc.metadata["document_type"] == "synthetic_log"),
        None,
    )
    if log_example is not None:
        print(json.dumps(log_example.metadata, indent=2))

    if args.export:
        export_path = PROJECT_FOLDER / "work" / "loaded_documents.jsonl"
        export_path.parent.mkdir(exist_ok=True)
        with export_path.open("w", encoding="utf-8") as file:
            for doc in documents:
                record = {"id": doc.id, "page_content": doc.page_content, "metadata": doc.metadata}
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\nLocal preview saved to: {export_path}")

    print("\nNo documents were sent to an AI provider or Pinecone.")


if __name__ == "__main__":
    main()
