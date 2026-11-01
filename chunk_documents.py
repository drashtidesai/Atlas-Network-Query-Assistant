"""Lesson 3: preview and optionally export structure-aware corpus chunks.

Run: uv run python chunk_documents.py
Optional local export: uv run python chunk_documents.py --export
All checks finish before export. No AI provider or Pinecone calls happen here.
"""

import argparse
import csv
import io
import json
import re

from langchain_core.documents import Document

from load_documents import PROJECT_FOLDER, load_documents, synthetic_log_metadata


def chunk_interfaces(document: Document) -> list[Document]:
    """Keep each interface heading and its indented commands together."""
    lines = document.page_content.splitlines()
    chunks = []

    for index, line in enumerate(lines):
        if not line.startswith("interface "):
            continue

        # Python indexes start at zero; end is the exclusive slice boundary.
        end = index + 1
        while end < len(lines) and lines[end].startswith(" "):
            end += 1

        # Copy source metadata so the original Document stays unchanged.
        metadata = document.metadata.copy()
        metadata.update(
            {
                "section": line,
                "line_start": document.metadata["line_start"] + index,
                "line_end": document.metadata["line_start"] + end - 1,
                "parent_id": document.id,
            }
        )

        chunk = Document(
            id=f"{document.id}:interface:{index + 1}-{end}",
            page_content="\n".join(lines[index:end]),
            metadata=metadata,
        )
        chunks.append(chunk)

    return chunks


def chunk_bgp(document: Document) -> list[Document]:
    """Keep each BGP heading and all its indented commands together."""
    lines = document.page_content.splitlines()
    chunks = []

    for index, line in enumerate(lines):
        if not line.startswith("router bgp "):
            continue

        # Both one-space and deeper indentation belong to the BGP block.
        end = index + 1
        while end < len(lines) and lines[end].startswith(" "):
            end += 1

        metadata = document.metadata.copy()
        metadata.update(
            {
                "section": line,
                "line_start": document.metadata["line_start"] + index,
                "line_end": document.metadata["line_start"] + end - 1,
                "parent_id": document.id,
            }
        )

        chunk = Document(
            id=f"{document.id}:bgp:{index + 1}-{end}",
            page_content="\n".join(lines[index:end]),
            metadata=metadata,
        )
        chunks.append(chunk)

    return chunks


def chunk_named_policy(document: Document, prefix: str, kind: str) -> list[Document]:
    """Group policy entries by name, retaining their original source order."""
    lines = document.page_content.splitlines()
    groups = {}

    for index, line in enumerate(lines):
        if not line.startswith(prefix):
            continue

        # After the prefix, the first word is the policy name.
        name = line[len(prefix):].split()[0]
        end = index + 1
        while end < len(lines) and lines[end].startswith(" "):
            end += 1

        if name not in groups:
            groups[name] = []
        groups[name].append((index, end))

    chunks = []
    for name, ranges in groups.items():
        content_lines = []
        source_line_ranges = []
        for start, end in ranges:
            content_lines.extend(lines[start:end])
            first_line = document.metadata["line_start"] + start
            last_line = document.metadata["line_start"] + end - 1
            source_line_ranges.append(f"{first_line}-{last_line}")

        metadata = document.metadata.copy()
        metadata.update(
            {
                "section": f"{prefix}{name}",
                "line_start": document.metadata["line_start"] + ranges[0][0],
                "line_end": document.metadata["line_start"] + ranges[-1][1] - 1,
                # Exact ranges remain accurate even when entries are separated.
                "source_line_ranges": source_line_ranges,
                "parent_id": document.id,
            }
        )
        chunks.append(
            Document(
                id=f"{document.id}:{kind}:{name}",
                page_content="\n".join(content_lines),
                metadata=metadata,
            )
        )

    return chunks


def chunk_route_maps(document: Document) -> list[Document]:
    """Keep all clauses of each named route-map together."""
    return chunk_named_policy(document, "route-map ", "route-map")


def chunk_prefix_lists(document: Document) -> list[Document]:
    """Keep all entries of each named IPv4 prefix-list together."""
    return chunk_named_policy(document, "ip prefix-list ", "prefix-list")


def chunk_indented_blocks(document: Document, prefix: str, kind: str) -> list[Document]:
    """Extract headings matching a prefix plus their indented commands."""
    lines = document.page_content.splitlines()
    chunks = []

    for index, line in enumerate(lines):
        if not line.startswith(prefix):
            continue

        end = index + 1
        while end < len(lines) and lines[end].startswith(" "):
            end += 1

        metadata = document.metadata.copy()
        metadata.update(
            {
                "section": line,
                "line_start": document.metadata["line_start"] + index,
                "line_end": document.metadata["line_start"] + end - 1,
                "parent_id": document.id,
            }
        )
        chunks.append(
            Document(
                id=f"{document.id}:{kind}:{index + 1}-{end}",
                page_content="\n".join(lines[index:end]),
                metadata=metadata,
            )
        )

    return chunks


def chunk_acls(document: Document) -> list[Document]:
    """Extract the named IPv4 ACL blocks used in this corpus."""
    return chunk_indented_blocks(document, "ip access-list ", "acl")


def chunk_vlans(document: Document) -> list[Document]:
    """Keep each VLAN definition and its name together."""
    return chunk_indented_blocks(document, "vlan ", "vlan")


def chunk_terminal_lines(document: Document) -> list[Document]:
    """Keep console and VTY blocks with their access settings."""
    return chunk_indented_blocks(document, "line ", "terminal-line")


def chunk_tacacs_servers(document: Document) -> list[Document]:
    """Keep each TACACS server definition and its settings together."""
    return chunk_indented_blocks(document, "tacacs server ", "tacacs-server")


def chunk_tacacs_groups(document: Document) -> list[Document]:
    """Keep each TACACS server group and its member names together."""
    return chunk_indented_blocks(document, "aaa group server tacacs+ ", "tacacs-group")


def chunk_source_lines(chunk: Document) -> list[int]:
    """Read exact source line numbers from a chunk's metadata."""
    ranges = chunk.metadata.get("source_line_ranges")
    if ranges is None:
        ranges = [f"{chunk.metadata['line_start']}-{chunk.metadata['line_end']}"]

    numbers = []
    for source_range in ranges:
        first, last = source_range.split("-")
        numbers.extend(range(int(first), int(last) + 1))
    return numbers


def chunk_remaining_settings(document: Document, structured: list[Document]) -> list[Document]:
    """Collect consecutive uncovered lines, preserving comments with settings."""
    covered = set()
    for chunk in structured:
        covered.update(chunk_source_lines(chunk))

    lines = document.page_content.splitlines()
    first_line = document.metadata["line_start"]
    chunks = []
    index = 0
    while index < len(lines):
        text = lines[index].strip()
        if first_line + index in covered or text in {"", "!", "end"}:
            index += 1
            continue

        start = index
        while index < len(lines):
            text = lines[index].strip()
            if first_line + index in covered or text in {"", "!", "end"}:
                break
            index += 1

        metadata = document.metadata.copy()
        metadata.update(
            {
                "section": "general settings",
                "line_start": first_line + start,
                "line_end": first_line + index - 1,
                "parent_id": document.id,
            }
        )
        chunks.append(
            Document(
                id=f"{document.id}:general:{start + 1}-{index}",
                page_content="\n".join(lines[start:index]),
                metadata=metadata,
            )
        )
    return chunks


def chunk_configuration(document: Document) -> list[Document]:
    """Combine our section extractors and collect the remaining settings."""
    if document.metadata["document_type"] not in {"intended_config", "running_config"}:
        raise ValueError("Configuration chunking requires a configuration Document.")

    structured = (
        chunk_interfaces(document)
        + chunk_bgp(document)
        + chunk_route_maps(document)
        + chunk_prefix_lists(document)
        + chunk_acls(document)
        + chunk_vlans(document)
        + chunk_terminal_lines(document)
        + chunk_tacacs_servers(document)
        + chunk_tacacs_groups(document)
    )
    chunks = structured + chunk_remaining_settings(document, structured)
    return sorted(chunks, key=lambda chunk: chunk.metadata["line_start"])


def report_coverage(document: Document, chunks: list[Document]) -> None:
    """Check command coverage and verify chunk text against source lines."""
    lines = document.page_content.splitlines()
    first_line = document.metadata["line_start"]
    counts = {}
    for chunk in chunks:
        if chunk.metadata["parent_id"] != document.id:
            raise ValueError("Coverage received a chunk from a different Document.")
        numbers = chunk_source_lines(chunk)
        if any(number < first_line or number >= first_line + len(lines) for number in numbers):
            raise ValueError("Chunk source range is outside the Document.")
        expected_text = "\n".join(lines[number - first_line] for number in numbers)
        if chunk.page_content != expected_text:
            raise ValueError(f"Chunk text differs from its source: {chunk.id}")
        for number in numbers:
            counts[number] = counts.get(number, 0) + 1

    # Comments, blank lines, and the final 'end' marker are not commands here.
    command_lines = {
        number
        for number, line in enumerate(lines, start=first_line)
        if line.strip() and not line.lstrip().startswith("!") and line.strip() != "end"
    }
    missing = sorted(number for number in command_lines if number not in counts)
    duplicated = sorted(number for number in command_lines if counts.get(number, 0) > 1)
    print(f"\nCoverage for {document.metadata['filename']}:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Command lines covered: {len(command_lines) - len(missing)}/{len(command_lines)}")
    print(f"  Missing command lines: {missing}")
    print(f"  Duplicated command lines: {duplicated}")
    print("  Chunk text matches its source ranges.")
    if missing or duplicated:
        raise ValueError("Configuration coverage needs attention.")


def check_source_metadata(document: Document, chunks: list[Document]) -> None:
    """Check that source labels survive unchanged on every child chunk."""
    if not chunks:
        raise ValueError(f"No chunks created for {document.metadata['filename']}.")

    for chunk in chunks:
        for key, value in document.metadata.items():
            # Chunk line bounds describe the excerpt, not the whole source.
            if key in {"line_start", "line_end"}:
                continue
            if key not in chunk.metadata or chunk.metadata[key] != value:
                raise ValueError(f"Source metadata changed: {key} on {chunk.id}")

        if document.metadata["document_type"] == "running_config":
            if not chunk.metadata.get("snapshot_at"):
                raise ValueError("Running-config chunk is missing its timestamp.")
            if chunk.metadata.get("snapshot_is_simulated") is not True:
                raise ValueError("This fictional snapshot must retain its simulated flag.")

    print("  Source metadata preserved on every chunk.")
    if document.metadata["document_type"] == "running_config":
        print(f"  Snapshot timestamp preserved: {document.metadata['snapshot_at']}")
        print("  Simulated flag preserved: true")


def chunk_prose(document: Document) -> list[Document]:
    """Split prose sections, keeping headings and paragraphs intact."""
    if document.metadata["document_type"] not in {"design_document", "technology_reference"}:
        raise ValueError("Prose chunking requires a design document or reference note.")

    lines = document.page_content.splitlines()
    headings = [
        index for index, line in enumerate(lines)
        if re.match(r"^\d+\.\s+", line) or line == "Read-only evidence commands"
    ]
    # Include the introduction as its own chunk, before the first heading.
    boundaries = sorted(set([0] + headings + [len(lines)]))
    introduction_end = headings[0] if headings else len(lines)
    context = "\n".join(lines[:introduction_end]).rstrip()
    chunks = []

    for start, end in zip(boundaries, boundaries[1:]):
        # Remove boundary blank lines while retaining internal paragraph breaks.
        while start < end and not lines[start].strip():
            start += 1
        while end > start and not lines[end - 1].strip():
            end -= 1
        if start == end:
            continue

        metadata = document.metadata.copy()
        metadata.update(
            {
                "section": lines[start] if start in headings else "introduction",
                "line_start": document.metadata["line_start"] + start,
                "line_end": document.metadata["line_start"] + end - 1,
                "parent_id": document.id,
                # Keep document-level qualifications available with every section.
                "document_context": context,
            }
        )
        chunks.append(
            Document(
                id=f"{document.id}:prose:{start + 1}-{end}",
                page_content="\n".join(lines[start:end]),
                metadata=metadata,
            )
        )
    return chunks


def report_prose_coverage(document: Document, chunks: list[Document]) -> None:
    """Verify exact text and one-time coverage of every nonblank prose line."""
    lines = document.page_content.splitlines()
    first_line = document.metadata["line_start"]
    counts = {}
    for chunk in chunks:
        if chunk.metadata["parent_id"] != document.id:
            raise ValueError("Prose chunk belongs to a different Document.")
        numbers = chunk_source_lines(chunk)
        if any(number < first_line or number >= first_line + len(lines) for number in numbers):
            raise ValueError("Prose chunk range is outside its source.")
        expected = "\n".join(lines[number - first_line] for number in numbers)
        if chunk.page_content != expected:
            raise ValueError("Prose chunk text differs from its source range.")
        for number in numbers:
            counts[number] = counts.get(number, 0) + 1

    required = {
        number for number, line in enumerate(lines, start=first_line)
        if line.strip()
    }
    missing = sorted(number for number in required if number not in counts)
    duplicated = sorted(number for number in required if counts.get(number, 0) > 1)
    print(f"\nProse coverage for {document.metadata['filename']}:")
    print(f"  Nonblank lines covered: {len(required) - len(missing)}/{len(required)}")
    print(f"  Missing prose lines: {missing}")
    print(f"  Duplicated prose lines: {duplicated}")
    print("  Chunk text matches its source ranges.")
    if missing or duplicated:
        raise ValueError("Prose coverage needs attention.")
    check_source_metadata(document, chunks)


def check_prose_context(document: Document, chunks: list[Document]) -> None:
    """Check introduction metadata and reference URL placement."""
    introduction = next(
        (chunk for chunk in chunks if chunk.metadata["section"] == "introduction"),
        None,
    )
    if introduction is None:
        raise ValueError("These prose sources should have a document introduction.")
    for chunk in chunks:
        if chunk.metadata.get("document_context") != introduction.page_content:
            raise ValueError("Document introduction was not preserved in chunk context.")
    print("  Document introduction preserved in every chunk's context metadata.")

    if document.metadata["document_type"] == "technology_reference":
        current_heading = None
        url_count = 0
        for number, line in enumerate(
            document.page_content.splitlines(), start=document.metadata["line_start"]
        ):
            if re.match(r"^\d+\.\s+", line):
                current_heading = line
            if line.startswith(("https://", "http://")):
                owners = [
                    chunk for chunk in chunks
                    if chunk.metadata["line_start"] <= number <= chunk.metadata["line_end"]
                ]
                if len(owners) != 1 or owners[0].metadata["section"] != current_heading:
                    raise ValueError("Reference URL was separated from its numbered note.")
                url_count += 1
        print(f"  Reference URLs kept with their numbered notes: {url_count}.")

    if document.metadata["filename"] == "failover-and-test-plan.txt":
        commands = [
            chunk for chunk in chunks
            if chunk.metadata["section"] == "Read-only evidence commands"
        ]
        if len(commands) != 1:
            raise ValueError("Expected a separate read-only evidence commands section.")
        print("  Read-only evidence commands preserved as a separate section.")


def chunk_csv(document: Document) -> list[Document]:
    """Turn each CSV data record into labelled text, retaining empty cells."""
    if document.metadata["file_format"] != "csv":
        raise ValueError("CSV chunking requires a CSV Document.")

    reader = csv.reader(io.StringIO(document.page_content, newline=""), strict=True)
    headers = next(reader)
    if not headers or any(not header.strip() for header in headers) or len(set(headers)) != len(headers):
        raise ValueError("CSV column names must be nonempty and unique.")

    first_line = document.metadata["line_start"]
    header_end = first_line + reader.line_num - 1
    chunks = []
    for row_number, values in enumerate(reader, start=1):
        # reader.line_num counts physical lines, including quoted multiline cells.
        line_start = header_end + 1 if row_number == 1 else previous_end + 1
        line_end = first_line + reader.line_num - 1
        previous_end = line_end
        if len(values) != len(headers):
            raise ValueError(f"CSV row {row_number} has the wrong number of cells.")

        metadata = document.metadata.copy()
        metadata.update(
            {
                "section": "CSV data row",
                "row_number": row_number,
                "line_start": line_start,
                "line_end": line_end,
                "header_line_start": first_line,
                "header_line_end": header_end,
                "parent_id": document.id,
            }
        )
        chunks.append(
            Document(
                id=f"{document.id}:csv-row:{row_number}",
                page_content="\n".join(
                    f"{header}: {value}" for header, value in zip(headers, values)
                ),
                metadata=metadata,
            )
        )
    return chunks


def report_csv_coverage(document: Document, chunks: list[Document]) -> None:
    """Check each labelled record against the original parsed CSV values."""
    reader = csv.reader(io.StringIO(document.page_content, newline=""), strict=True)
    headers = next(reader)
    header_end = document.metadata["line_start"] + reader.line_num - 1
    previous_end = header_end
    checked = 0
    for row_number, values in enumerate(reader, start=1):
        if row_number > len(chunks):
            raise ValueError("A CSV data row is missing its chunk.")
        chunk = chunks[row_number - 1]
        line_end = document.metadata["line_start"] + reader.line_num - 1
        expected = "\n".join(f"{header}: {value}" for header, value in zip(headers, values))
        if len(values) != len(headers) or chunk.page_content != expected:
            raise ValueError(f"CSV values changed on data row {row_number}.")
        if (
            chunk.metadata["parent_id"] != document.id
            or chunk.metadata["row_number"] != row_number
            or chunk.metadata["line_start"] != previous_end + 1
            or chunk.metadata["line_end"] != line_end
            or chunk.metadata["header_line_start"] != document.metadata["line_start"]
            or chunk.metadata["header_line_end"] != header_end
        ):
            raise ValueError("CSV chunk provenance does not match its source record.")
        previous_end = line_end
        checked += 1
    if checked != len(chunks):
        raise ValueError("Unexpected extra CSV chunks were found.")
    print(f"\nCSV coverage for {document.metadata['filename']}:")
    print(f"  Data rows preserved: {checked}/{checked}")
    print("  Every column value, including empty cells, matches the source.")
    print("  Data-row numbers and source/header line bounds match the source.")
    check_source_metadata(document, chunks)


def chunk_ascii_topology(document: Document) -> list[Document]:
    """Preserve the complete ASCII diagram and notes in one chunk."""
    if document.metadata["source"] != "corpus/topology/topology-ascii.txt":
        raise ValueError("Expected the ASCII topology source.")
    metadata = document.metadata.copy()
    metadata.update({"section": "complete ASCII topology", "parent_id": document.id})
    return [
        Document(
            id=f"{document.id}:ascii-topology",
            page_content=document.page_content,
            metadata=metadata,
        )
    ]


def chunk_synthetic_log(document: Document) -> list[Document]:
    """Keep timelines intact, splitting FAIL-08 at service baseline boundaries."""
    if document.metadata["document_type"] != "synthetic_log":
        raise ValueError("Log chunking requires a synthetic-log Document.")

    if document.metadata["scenario_id"] == "FAIL-08":
        lines = document.page_content.splitlines()
        # Start each service at its baseline, not at the later fault injection.
        markers = ("phase=pre_08D ", "phase=pre_08E ", "phase=pre_08F ")
        starts = [3]
        for marker in markers:
            matches = [index for index, line in enumerate(lines) if marker in line]
            if not matches:
                raise ValueError(f"Missing management-service baseline: {marker}")
            starts.append(matches[0])
        if starts != sorted(set(starts)):
            raise ValueError("Management service boundaries are not in source order.")

        sections = (
            ("tacacs-console", "08A-08C TACACS and console"),
            ("syslog", "08D Syslog"),
            ("snmp", "08E SNMP"),
            ("ntp", "08F NTP"),
        )
        chunks = []
        for (service, section), start, end in zip(sections, starts, starts[1:] + [len(lines)]):
            # Repeat only the three non-event header lines for provenance.
            content = "\n".join(lines[:3] + lines[start:end])
            timeline = synthetic_log_metadata(content, document.metadata["filename"])
            first_line = document.metadata["line_start"]
            metadata = document.metadata.copy()
            metadata.update(
                {
                    "section": section,
                    "parent_id": document.id,
                    "line_start": first_line,
                    "line_end": first_line + end - 1,
                    "source_line_ranges": [
                        f"{first_line}-{first_line + 2}",
                        f"{first_line + start}-{first_line + end - 1}",
                    ],
                    "shared_header_line_count": 3,
                    # Original event_* fields still describe the whole source.
                    "chunk_event_count": timeline["event_count"],
                    "chunk_event_time_start": timeline["event_time_start"],
                    "chunk_event_time_end": timeline["event_time_end"],
                    "chunk_observers": timeline["observers"],
                }
            )
            chunks.append(Document(
                id=f"{document.id}:synthetic-service:{service}",
                page_content=content,
                metadata=metadata,
            ))
        return chunks

    metadata = document.metadata.copy()
    metadata.update(
        {"section": "complete synthetic scenario timeline", "parent_id": document.id}
    )
    return [
        Document(
            id=f"{document.id}:synthetic-timeline",
            page_content=document.page_content,
            metadata=metadata,
        )
    ]


def report_log_coverage(document: Document, chunks: list[Document]) -> None:
    """Verify exact text; only shared non-event headers may be repeated."""
    expected_count = 4 if document.metadata["scenario_id"] == "FAIL-08" else 1
    if len(chunks) != expected_count:
        raise ValueError("Unexpected number of scenario/service chunks.")
    lines = document.page_content.splitlines()
    first_line = document.metadata["line_start"]
    expected_lines = list(range(
        document.metadata["line_start"], document.metadata["line_end"] + 1
    ))
    counts = {}
    total_events = 0
    previous_event_end = None
    for chunk in chunks:
        if chunk.metadata["parent_id"] != document.id:
            raise ValueError("Log chunk belongs to a different source Document.")
        numbers = chunk_source_lines(chunk)
        if any(number not in expected_lines for number in numbers) or numbers != sorted(set(numbers)):
            raise ValueError("Invalid or unordered source ranges in log chunk.")
        expected_text = "\n".join(lines[number - first_line] for number in numbers)
        if chunk.page_content != expected_text:
            raise ValueError("Log chunk text differs from its precise source ranges.")
        if chunk.page_content.splitlines()[:3] != lines[:3]:
            raise ValueError("Scenario context was lost from a log chunk.")
        timeline = synthetic_log_metadata(chunk.page_content, document.metadata["filename"])
        for key, value in timeline.items():
            chunk_key = f"chunk_{key}" if len(chunks) > 1 and key in {
                "event_count", "event_time_start", "event_time_end", "observers"
            } else key
            if chunk.metadata.get(chunk_key) != value:
                raise ValueError(f"Log metadata does not match chunk text: {chunk_key}")
        if previous_event_end is not None and timeline["event_time_start"] < previous_event_end:
            raise ValueError("Service chunks are not in chronological order.")
        previous_event_end = timeline["event_time_end"]
        total_events += timeline["event_count"]
        for number in numbers:
            counts[number] = counts.get(number, 0) + 1

    for number in expected_lines:
        expected_occurrences = len(chunks) if number < first_line + 3 else 1
        if counts.get(number, 0) != expected_occurrences:
            raise ValueError("Missing source line or unexpected duplicated event.")
    if total_events != document.metadata["event_count"]:
        raise ValueError("Total chunk event count differs from the original timeline.")

    print(f"\nLog coverage for {document.metadata['filename']}:")
    print(f"  Scenario: {document.metadata['scenario_id']} | chunks: {len(chunks)}")
    print(f"  Source lines preserved: {len(expected_lines)}/{len(expected_lines)}")
    print(f"  Event records preserved exactly once: {total_events}")
    print("  Exact source text, simulation labels and chronological order preserved.")
    if len(chunks) > 1:
        print("  Only the three non-event context lines are repeated per service chunk.")
        for chunk in chunks:
            print(f"    {chunk.metadata['section']}: {chunk.metadata['chunk_event_count']} events; source ranges={chunk.metadata['source_line_ranges']}")
    print(f"  Source UTC range: {document.metadata['event_time_start']} -> {document.metadata['event_time_end']}")
    check_source_metadata(document, chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export",
        action="store_true",
        help="Save checked chunks to work/chunks.jsonl, replacing any previous export.",
    )
    args = parser.parse_args()

    documents = load_documents()
    document = next(
        doc for doc in documents
        if doc.metadata["source"] == "corpus/intended-configs/DXB-R01.txt"
    )
    chunks = chunk_interfaces(document)
    example = next(
        chunk for chunk in chunks
        if chunk.metadata["section"] == "interface GigabitEthernet0/0/0"
    )

    print(f"Created {len(chunks)} interface chunks from {document.metadata['filename']}.")
    print("\nExample chunk text:")
    print(example.page_content)
    print("\nExample chunk metadata:")
    print(json.dumps(example.metadata, indent=2))
    bgp_chunks = chunk_bgp(document)
    print(f"\nCreated {len(bgp_chunks)} BGP chunks from {document.metadata['filename']}.")
    for bgp_chunk in bgp_chunks:
        print("\nBGP chunk text:")
        print(bgp_chunk.page_content)
        print("\nBGP chunk metadata:")
        print(json.dumps(bgp_chunk.metadata, indent=2))

    route_map_chunks = chunk_route_maps(document)
    prefix_list_chunks = chunk_prefix_lists(document)
    print(f"\nCreated {len(route_map_chunks)} route-map chunks from {document.metadata['filename']}.")
    print(f"Created {len(prefix_list_chunks)} prefix-list chunks from {document.metadata['filename']}.")

    for policy_chunk in route_map_chunks + prefix_list_chunks:
        if policy_chunk.metadata["section"] in {"route-map WAN-IN", "ip prefix-list REMOTE-SITE"}:
            print("\nPolicy chunk text:")
            print(policy_chunk.page_content)
            print("\nPolicy chunk metadata:")
            print(json.dumps(policy_chunk.metadata, indent=2))

    switch_document = next(
        doc for doc in documents
        if doc.metadata["source"] == "corpus/intended-configs/DXB-SW01.txt"
    )
    acl_chunks = chunk_acls(document)
    terminal_chunks = chunk_terminal_lines(document)
    vlan_chunks = chunk_vlans(switch_document)

    print(f"\nCreated {len(acl_chunks)} ACL chunks from {document.metadata['filename']}.")
    print(f"Created {len(terminal_chunks)} terminal-line chunks from {document.metadata['filename']}.")
    print(f"Created {len(vlan_chunks)} VLAN chunks from {switch_document.metadata['filename']}.")

    for block_chunk in acl_chunks + terminal_chunks + vlan_chunks:
        if block_chunk.metadata["section"] in {
            "ip access-list standard ADMIN-SOURCES",
            "line vty 0 15",
            "vlan 110",
        }:
            print("\nConfiguration block text:")
            print(block_chunk.page_content)
            print("\nConfiguration block metadata:")
            print(json.dumps(block_chunk.metadata, indent=2))

    tacacs_server_chunks = chunk_tacacs_servers(document)
    tacacs_group_chunks = chunk_tacacs_groups(document)
    print(f"\nCreated {len(tacacs_server_chunks)} TACACS server chunks from {document.metadata['filename']}.")
    print(f"Created {len(tacacs_group_chunks)} TACACS group chunks from {document.metadata['filename']}.")

    for tacacs_chunk in tacacs_server_chunks + tacacs_group_chunks:
        print("\nTACACS chunk text:")
        print(tacacs_chunk.page_content)
        print("\nTACACS chunk metadata:")
        print(json.dumps(tacacs_chunk.metadata, indent=2))

    for config_document in (document, switch_document):
        config_chunks = chunk_configuration(config_document)
        general_chunks = [
            chunk for chunk in config_chunks
            if chunk.metadata["section"] == "general settings"
        ]
        print(f"\nCreated {len(general_chunks)} general-settings chunks from {config_document.metadata['filename']}.")
        if config_document is document:
            for general_chunk in general_chunks:
                print("\nGeneral-settings chunk text:")
                print(general_chunk.page_content)
                print("\nGeneral-settings chunk metadata:")
                print(json.dumps(general_chunk.metadata, indent=2))
    configuration_documents = [
        doc for doc in documents
        if doc.metadata["document_type"] in {"intended_config", "running_config"}
    ]
    all_config_chunks = []
    print("\nChecking every configuration file:")
    for config_document in configuration_documents:
        config_chunks = chunk_configuration(config_document)
        report_coverage(config_document, config_chunks)
        check_source_metadata(config_document, config_chunks)
        all_config_chunks.extend(config_chunks)

    chunk_ids = [chunk.id for chunk in all_config_chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate configuration chunk IDs were found.")

    print(f"\nChecked {len(configuration_documents)} configuration files; created {len(all_config_chunks)} chunks.")
    print("All configuration chunk IDs are unique.")
    lld_document = next(
        doc for doc in documents
        if doc.metadata["source"] == "corpus/design/low-level-design.txt"
    )
    prose_chunks = chunk_prose(lld_document)
    print(f"\nCreated {len(prose_chunks)} prose chunks from {lld_document.metadata['filename']}.")
    vrrp_chunk = next(chunk for chunk in prose_chunks if chunk.metadata["section"] == "3. VRRP")
    print("\nVRRP prose chunk text:")
    print(vrrp_chunk.page_content)
    print("\nVRRP prose chunk metadata:")
    print(json.dumps(vrrp_chunk.metadata, indent=2))
    prose_documents = [
        doc for doc in documents
        if doc.metadata["document_type"] in {"design_document", "technology_reference"}
    ]
    all_prose_chunks = []
    print("\nChecking every design and reference file:")
    for prose_document in prose_documents:
        file_chunks = chunk_prose(prose_document)
        print(f"\nCreated {len(file_chunks)} prose chunks from {prose_document.metadata['filename']}.")
        report_prose_coverage(prose_document, file_chunks)
        check_prose_context(prose_document, file_chunks)
        all_prose_chunks.extend(file_chunks)

    all_ids = [chunk.id for chunk in all_config_chunks + all_prose_chunks]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate chunk IDs were found across configurations and prose.")
    print(f"\nChecked {len(prose_documents)} design/reference files; created {len(all_prose_chunks)} prose chunks.")
    print("All configuration and prose chunk IDs are unique.")
    devices_document = next(
        doc for doc in documents
        if doc.metadata["source"] == "corpus/topology/devices.csv"
    )
    device_chunks = chunk_csv(devices_document)
    print(f"\nCreated {len(device_chunks)} CSV chunks from devices.csv.")
    print("\nFirst device row text:")
    print(device_chunks[0].page_content)
    print("\nFirst device row metadata:")
    print(json.dumps(device_chunks[0].metadata, indent=2))
    csv_documents = [doc for doc in documents if doc.metadata["file_format"] == "csv"]
    all_csv_chunks = []
    print("\nChecking every CSV file:")
    for csv_document in csv_documents:
        row_chunks = chunk_csv(csv_document)
        report_csv_coverage(csv_document, row_chunks)
        all_csv_chunks.extend(row_chunks)

    topology_document = next(
        doc for doc in documents
        if doc.metadata["source"] == "corpus/topology/topology-ascii.txt"
    )
    topology_chunks = chunk_ascii_topology(topology_document)
    if topology_chunks[0].page_content != topology_document.page_content:
        raise ValueError("ASCII topology text or layout changed.")
    if chunk_source_lines(topology_chunks[0]) != list(range(
        topology_document.metadata["line_start"], topology_document.metadata["line_end"] + 1
    )):
        raise ValueError("ASCII topology source line bounds changed.")
    print("\nASCII topology: 1 chunk; complete loaded text and layout preserved.")
    check_source_metadata(topology_document, topology_chunks)

    log_documents = [
        doc for doc in documents if doc.metadata["document_type"] == "synthetic_log"
    ]
    all_log_chunks = []
    print("\nChecking every synthetic log file:")
    for log_document in log_documents:
        log_chunks = chunk_synthetic_log(log_document)
        report_log_coverage(log_document, log_chunks)
        all_log_chunks.extend(log_chunks)

    log_scenarios = [doc.metadata["scenario_id"] for doc in log_documents]
    if len(log_scenarios) != len(set(log_scenarios)):
        raise ValueError("Duplicate synthetic scenario IDs were found.")

    all_chunks = (
        all_config_chunks + all_prose_chunks + all_csv_chunks
        + topology_chunks + all_log_chunks
    )
    combined_ids = [chunk.id for chunk in all_chunks]
    if len(combined_ids) != len(set(combined_ids)):
        raise ValueError("Duplicate chunk IDs were found.")
    expected_sources = {doc.metadata["source"] for doc in documents}
    represented_sources = {chunk.metadata["source"] for chunk in all_chunks}
    if represented_sources != expected_sources:
        raise ValueError("Chunk sources do not match the loaded corpus sources.")
    print("\nComplete corpus chunking summary:")
    print(f"  Source files represented: {len(represented_sources)}/{len(documents)}")
    print(f"  Configuration chunks: {len(all_config_chunks)}")
    print(f"  Prose chunks: {len(all_prose_chunks)}")
    print(f"  CSV chunks: {len(all_csv_chunks)}")
    print(f"  ASCII topology chunks: {len(topology_chunks)}")
    print(f"  Synthetic log chunks: {len(all_log_chunks)}")
    print(f"  Total chunks: {len(all_chunks)}")
    print("  All chunk IDs are unique.")
    if args.export:
        export_path = PROJECT_FOLDER / "work" / "chunks.jsonl"
        export_path.parent.mkdir(exist_ok=True)
        with export_path.open("w", encoding="utf-8") as file:
            for chunk in all_chunks:
                record = {
                    "id": chunk.id,
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata,
                }
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\nExported {len(all_chunks)} chunks to: {export_path}")
    else:
        print("Preview only: chunks remain in memory for this run. No files were exported.")
    print("No AI provider or Pinecone was called.")


if __name__ == "__main__":
    main()
