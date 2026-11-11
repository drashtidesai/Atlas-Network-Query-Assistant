"""Typed, source-linked BGP facts. No inference of live session state."""
import re


def local_as(text):
    return [{"value": int(match[1]), "line": number}
            for number, line in enumerate(text.splitlines(), 1)
            if (match := re.fullmatch(r"router bgp (\d+)", line.strip()))]


def register(state, source, text, number):
    identifier = f"{source}:L{number}"
    state.setdefault("specialist_evidence", {})[identifier] = {
        "source": source, "line": number, "text": text.splitlines()[number - 1]}
    document = state.get("corpus_snapshots", {}).get(source)
    if document:
        state["specialist_evidence"][identifier].update(
            document_type=document["document_type"], sha256=document["sha256"], evidence_is_simulated=False)
        state.setdefault("corpus_accessed", {})[source] = document["sha256"]
    return identifier


def collect_bgp_facts(state, owner=None):
    owner = owner or state
    router = owner["router"]
    prefix = router + "/" if state.get("bundle_devices") else ""
    before, after = local_as(owner["baseline"]), local_as(owner["proposal"])
    local_ids = [register(state, prefix + source, owner[source], item["line"])
                 for source, entries in (("baseline", before), ("proposal", after)) for item in entries]
    changes = []
    for finding in owner["checks"].get("bgp", []):
        peer = finding["setting"].rsplit(" | ", 1)[-1]
        ids = [register(state, prefix + source, owner[source], finding[source]["line"])
               for source in ("baseline", "proposal") if finding.get(source)]
        changes.append({"field": "neighbor_remote_as", "neighbor": peer,
                        "baseline_remote_as": (finding.get("baseline") or {}).get("value"),
                        "proposed_remote_as": (finding.get("proposal") or {}).get("value"),
                        "evidence_ids": ids, "documented_peers": []})
        candidates = [(name, item["proposal"]) for name, item in state.get("bundle_devices", {}).items() if name != router]
        candidates = [(name + "/proposal", text) for name, text in candidates]
        proposed_devices = set(state.get("bundle_devices", {}))
        for source, document in state.get("corpus_snapshots", {}).items():
            if "/intended-configs/" in source and source.rsplit("/", 1)[-1].removesuffix(".txt") not in proposed_devices:
                candidates.append((source, document["text"]))
        for source, text in candidates:
            address_lines = [number for number, line in enumerate(text.splitlines(), 1)
                             if re.match(r"\s+ip address " + re.escape(peer) + r"\s", line)]
            if not address_lines:
                continue
            entries = local_as(text)
            peer_ids = [register(state, source, text, number)
                        for number in address_lines + [item["line"] for item in entries]]
            changes[-1]["documented_peers"].append({
                "source": source, "local_as_values": [item["value"] for item in entries],
                "evidence_ids": peer_ids,
                "proposed_remote_as_matches": (entries[0]["value"] == changes[-1]["proposed_remote_as"]) if len(entries) == 1 else None,
                "scope": "Documentary address/AS lookup; not live state or complete BGP compatibility validation."})
    fact = {"device": router, "baseline_local_as": [item["value"] for item in before],
            "proposed_local_as": [item["value"] for item in after],
            "local_as_changed": [item["value"] for item in before] != [item["value"] for item in after],
            "local_as_evidence_ids": local_ids, "neighbor_changes": changes,
            "instruction": "Neighbor remote-AS is the expected peer AS, not this device's local AS. Do not infer changes to other neighbors or policies from this entry alone. Restore the changed neighbor entry for rollback, not an unchanged router bgp process."}
    state.setdefault("bgp_facts", {})[router] = fact
    return fact


def required_bgp_evidence(state):
    required = set()
    for fact in state.get("bgp_facts", {}).values():
        required.update(fact["local_as_evidence_ids"])
        for change in fact["neighbor_changes"]:
            required.update(change["evidence_ids"])
            for peer in change["documented_peers"]:
                required.update(peer["evidence_ids"])
    return required
