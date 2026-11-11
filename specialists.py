"""Role-specific tools and structured, read-only specialist handoffs."""
import csv
import io
import re

from review_tools import schema, fingerprint

ROLE_PROMPTS = {
    "configuration": "Investigate actual configuration changes with the supplied tools. For a device bundle, inspect_proposal checks all devices; cover all device findings in the handoff. submit_review hands your findings to the Impact Investigator; it does not request final approval in three-agent mode.",
    "impact": """You are the Impact Investigator. Use find_references and lookup_topology to
investigate dependencies of the changed settings, then submit_impact. Your only
job is documented potential impact, affected dependencies and explicit unknowns.
You cannot validate live state, execute commands or certify reachability. Cite
evidence IDs from tool results. Do not infer operator intent. Separate facts
from conditional impact hypotheses. Scope filter effects to the named neighbor,
not all WAN connectivity. Literal searches cannot prove complete impact coverage.
An interface address is not a subnet or virtual gateway. Never claim a confirmed
outage or that impact is confined to one protocol. Incorporate the latest human
revision note into your assessment and unknowns before handing off again.
Ask a human when essential intent is missing.""",
    "plan": """You are the Change Plan Agent. Inspect the structured configuration and
impact handoffs. The specialist_evidence map already contains exact source text.
Use get_evidence with evidence_ids to retrieve known IDs; inspect_change searches
literal configuration text, not ID strings. Empty search matches do not mean a
source file is unavailable. Submit a concise report summary plus prechecks, verification objectives,
rollback objectives and stop conditions using submit_plan. These are proposed,
unexecuted steps for engineer review, never commands to execute. Do not generate
device command sequences or assume rollback is authorized. Cite evidence IDs;
never invent line numbers. State checker limits, do not claim syntax validation,
confirmed intent, measured recovery or deployment safety. Ask a human if needed.""",
}


def role_tools(role, base):
    from corpus_tools import CORPUS_TOOLS
    if role == "configuration":
        return base + CORPUS_TOOLS
    ask = [tool for tool in base if tool["function"]["name"] == "ask_human"]
    ask += [schema("get_evidence", "Retrieve numbered evidence lines from this review's saved configuration/design/topology/corpus snapshots. Previously unread lines can be resolved; short corpus filenames require a unique match. Use returned canonical IDs. No arbitrary file access.", {"evidence_ids": {"type": "array", "items": {"type": "string"}}})]
    ask += CORPUS_TOOLS
    if role == "impact":
        return ask + [tool for tool in base if tool["function"]["name"] == "read_design"] + [schema("find_references", "Find literal setting/object references in baseline and proposal with exact evidence IDs.", {"query": {"type": "string"}}),
                      schema("lookup_topology", "Find literal identifier in snapshotted CSV topology rows (selected router or changed prefix/interface).", {"query": {"type": "string"}}),
                      schema("submit_impact", "Hand documented potential impact and unknowns to the planning agent.", {"assessment": {"type": "string"}, "unknowns": {"type": "string"}, "evidence_ids": {"type": "array", "items": {"type": "string"}}})]
    return ask + [schema("inspect_change", "Read numbered baseline/proposal evidence matching a literal identifier.", {"query": {"type": "string"}}),
                  schema("submit_plan", "Submit proposed, unexecuted verification/rollback objectives for human approval.", {name: {"type": "string"} for name in ("summary", "prechecks", "verification", "rollback", "stop_conditions")} | {"evidence_ids": {"type": "array", "items": {"type": "string"}}})]


def remember(state, source, line, text):
    identifier = f"{source}:L{line}"
    state.setdefault("specialist_evidence", {})[identifier] = {"source": source, "line": line, "text": text}
    return {"id": identifier, "text": text}


def references(state, query):
    matches = []
    devices = state.get("bundle_devices", {state["router"]: state})
    for router, device in devices.items():
        for source in ("baseline", "proposal"):
            label = f"{router}/{source}" if state.get("bundle_devices") else source
            for number, line in enumerate(device[source].splitlines(), 1):
                if query.casefold() in line.casefold():
                    matches.append((label, number, line))
    total = len(matches)
    matches = [remember(state, source, number, line) for source, number, line in matches[:60]]
    return {"matches": matches, "truncated": total > 60,
            "scope": "Literal text lookup, not a complete dependency graph."}


def specialist_dispatch(state, name, args):
    role = state.get("agent_role", "configuration")
    allowed = {tool["function"]["name"]: tool for tool in role_tools(role, [])}
    if name not in allowed or not isinstance(args, dict):
        return {"error": "tool_not_allowed_for_role", "retryable": False}
    props = allowed[name]["function"]["parameters"]["properties"]
    if set(args) != set(props):
        return {"error": "invalid_specialist_arguments", "retryable": False}
    for key, value in args.items():
        if key == "evidence_ids":
            if not isinstance(value, list) or len(value) > 60 or any(not isinstance(x, str) for x in value):
                return {"error": "invalid_evidence_ids", "retryable": False}
        elif not isinstance(value, str) or not value.strip() or len(value) > 8000:
            return {"error": "invalid_specialist_value", "retryable": False}
    if name in {"find_references", "inspect_change"}:
        state["specialist_queries"] = state.get("specialist_queries", []) + [name]
        # Backward-compatible recovery for models that send IDs to the text tool.
        ids = re.findall(r"(?:[\w./-]+/)?(?:baseline|proposal|[\w.-]+\.(?:txt|csv)):L\d+", args["query"])
        if ids:
            return evidence_by_id(state, ids)
        return references(state, args["query"])
    if name == "get_evidence":
        return evidence_by_id(state, args["evidence_ids"])
    if name == "lookup_topology":
        found = []
        for source, text in state["topology_snapshots"].items():
            # DictReader.line_num preserves physical line provenance for CSV rows.
            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                if args["query"].casefold() in " ".join(row.values()).casefold():
                    found.append(remember(state, source, reader.line_num, str(row)))
        state["specialist_queries"] = state.get("specialist_queries", []) + [name]
        return {"matches": found[:40], "scope": "Fictional documented topology, not observed state."}
    ids = [state.get("evidence_aliases", {}).get(item, item) for item in args["evidence_ids"]]
    args = {**args, "evidence_ids": ids}
    missing = [item for item in ids if item not in state.get("specialist_evidence", {})]
    if not ids or missing:
        return {"error": "unknown_or_missing_evidence", "missing_ids": missing,
                "retryable": True, "instruction": "Retrieve missing IDs using get_evidence, then cite returned canonical IDs. Remove unsupported claims if evidence cannot be resolved; do not invent sources."}
    if state.get("bgp_facts"):
        from bgp_facts import required_bgp_evidence
        # Local-AS context is already verified by the typed extractor. Attach it
        # transparently rather than reject a model's otherwise supported neighbor
        # assessment solely for omitting these context IDs from its list.
        context_ids = {identifier for fact in state["bgp_facts"].values()
                       for identifier in fact["local_as_evidence_ids"]}
        missing = (required_bgp_evidence(state) - context_ids) - set(ids)
        if missing:
            return {"error": "bgp_configuration_evidence_required", "missing_ids": sorted(missing),
                    "retryable": True, "instruction": "Cite typed local-AS, changed remote-AS and documented peer evidence already supplied in bgp_facts. Do not infer a local-AS change from remote-as."}
        added = sorted(context_ids - set(ids))
        if added:
            if any(identifier not in state.get("specialist_evidence", {}) for identifier in added):
                return {"error": "required_context_evidence_unavailable", "retryable": False}
            ids = ids + added
            args = {**args, "evidence_ids": ids, "controller_added_context_ids": added}
    if state.get("bundle_devices"):
        missing_devices = [router for router in state["target_routers"]
                           if not any(identifier.startswith(router + "/") for identifier in ids)]
        if missing_devices:
            return {"error": "bundle_device_evidence_required", "devices": missing_devices,
                    "retryable": True,
                    "instruction": "Cite configuration evidence for every bundle device. IDs from inspect_proposal are already available; reuse them."}
    if name == "submit_impact":
        from review_policy import no_changes
        if not no_changes(state) and not {"find_references", "lookup_topology"} <= set(state.get("specialist_queries", [])):
            return {"error": "impact_lookups_required", "retryable": True}
        state["impact_assessment"] = args
        state["agent_role"] = "plan"
        state["role_reset"] = True
        return {"status": "handoff", "to": "plan"}
    state["change_plan"] = {key: value for key, value in args.items() if key != "summary"}
    from review_tools import make_report
    result = make_report(state, args["summary"], finalizing=True)
    if "report" in state:
        state["report"].update(impact_assessment=state["impact_assessment"],
                               change_plan=state["change_plan"],
                               specialist_evidence=state["specialist_evidence"],
                               architecture="three_agent", plan_execution="not_executed")
        state["report"]["evidence"]["topology"] = {name: {"sha256": fingerprint(text)}
                                                         for name, text in state["topology_snapshots"].items()}
        state["report"]["evidence"]["corpus_sources_used"] = state.get("corpus_accessed", {})
        state["report_hash"] = fingerprint(state["report"])
        result["report_hash"] = state["report_hash"]
    return result


def evidence_by_id(state, ids):
    evidence = state.get("specialist_evidence", {})
    matches, missing = [], []
    for requested in ids:
        key = state.get("evidence_aliases", {}).get(requested, requested)
        if key not in evidence:
            key = resolve_snapshot_line(state, key)
            evidence = state.get("specialist_evidence", {})
        if key in evidence:
            matches.append({"id": key, "requested_id": requested, **evidence[key]})
            if requested != key:
                state.setdefault("evidence_aliases", {})[requested] = key
        else:
            missing.append(requested)
    return {"matches": matches,
            "missing_ids": missing,
            "note": "Missing IDs are not proof of missing files. Use literal searches or report an evidence gap; never invent content."}


def resolve_snapshot_line(state, identifier):
    """Resolve a single numbered line only from this review's saved sources."""
    match = re.fullmatch(r"(.+):L([1-9]\d*)", identifier)
    if not match:
        return identifier
    source, number = match[1], int(match[2])
    text, metadata = None, {}
    if source in {"baseline", "proposal"} and not state.get("bundle_devices"):
        text = state.get(source)
    elif "/" in source:
        device, kind = source.split("/", 1)
        if device in state.get("bundle_devices", {}) and kind in {"baseline", "proposal"}:
            text = state["bundle_devices"][device][kind]
    if text is None and source == "low-level-design.txt":
        text = state.get("design")
    if text is None and source in state.get("topology_snapshots", {}):
        text = state["topology_snapshots"][source]
    if text is None:
        documents = state.get("corpus_snapshots", {})
        if source not in documents and "/" not in source:
            candidates = [name for name in documents if name.rsplit("/", 1)[-1] == source]
            if len(candidates) == 1:
                source = candidates[0]
        if source in documents:
            document = documents[source]
            text = document["text"]
            metadata = {key: value for key, value in document.items() if key not in {"text", "line_count"}}
    if text is None or number > len(text.splitlines()):
        return identifier
    record = remember(state, source, number, text.splitlines()[number - 1])
    state["specialist_evidence"][record["id"]].update(metadata)
    if source in state.get("corpus_snapshots", {}):
        state.setdefault("corpus_accessed", {})[source] = state["corpus_snapshots"][source]["sha256"]
    return record["id"]
