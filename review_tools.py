"""Read-only configuration tools and deterministic report prerequisites."""

import difflib
import hashlib
import json
import re
import ipaddress
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROUTERS = ("DXB-R01", "DXB-R02", "LON-R01", "LON-R02")
DOMAINS = ("bgp", "vrrp", "interfaces", "bfd", "prefix_lists")
LIMIT = "Offline consistency with the supplied baseline only; no device, complete IOS syntax or deployment safety validation."


def fingerprint(value):
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_resource(name):
    if name not in {*(router + ".txt" for router in ROUTERS), "low-level-design.txt"}:
        raise ValueError("Resource not allowed.")
    path = ROOT / "resources" / name
    return path.read_text(encoding="utf-8-sig") if path.is_file() else None


def create_state(router, proposal, request, fault="none"):
    if router not in ROUTERS or fault not in {"none", "transient_compare", "permanent_compare"}:
        raise ValueError("Unsupported router or failure mode.")
    if not isinstance(proposal, str) or not proposal.strip() or len(proposal.encode()) > 100_000:
        raise ValueError("Supply a nonempty proposal no larger than 100 KB.")
    if not isinstance(request, str) or len(request) > 3000:
        raise ValueError("Request must be text of at most 3000 characters.")
    baseline = read_resource(f"{router}.txt")
    design = read_resource("low-level-design.txt")
    hostname = re.search(r"^hostname (\S+)\s*$", proposal, re.M)
    if not hostname or hostname[1] != router:
        raise ValueError("Proposal must contain the selected router's hostname.")
    return {"router": router, "proposal": proposal, "request": request.strip(),
            "baseline": baseline, "design": design, "baseline_hash": fingerprint(baseline),
            "proposal_hash": fingerprint(proposal), "design_hash": fingerprint(design),
            "messages": [], "events": [], "checks": {}, "design_read": [],
            "status": "running", "steps": 0, "calls": 0, "model_errors": 0,
            "tool_attempts": {}, "tool_failures": {}, "fault": fault, "active_seconds": 0.0,
            "usage": {"input_tokens": 0, "output_tokens": 0}, "human_history": [],
            "unsupported_acknowledged": False, "no_tool_count": 0}


def contextual_lines(text):
    context = "global"
    for number, line in enumerate(text.splitlines(), 1):
        if line and not line[0].isspace() and not line.startswith("!"):
            context = line.strip()
        yield number, context, line.strip()


def supported_setting(context, line):
    """Recognize only explicit commands in their supported block context."""
    interface = context.startswith("interface ") and not context.startswith("interface range ")
    bgp = context.startswith("router bgp ")
    if bgp and re.fullmatch(r"neighbor \S+ remote-as \d+", line):
        return "bgp"
    if interface and re.fullmatch(r"vrrp \d+ priority \d+", line):
        return "vrrp"
    if interface and (re.fullmatch(r"ip address \S+ \S+", line)
                      or line in {"no ip address", "shutdown", "no shutdown"}
                      or re.fullmatch(r"encapsulation dot1Q \d+(?: native)?", line)):
        return "interfaces"
    if (interface and (re.fullmatch(r"bfd interval \d+ min_rx \d+ multiplier \d+", line)
                       or line in {"bfd echo", "no bfd echo"})) or (
            bgp and re.fullmatch(r"(?:no )?neighbor \S+ fall-over bfd", line)):
        return "bfd"
    if re.fullmatch(r"ip prefix-list \S+ seq \d+ (?:permit|deny) \S+(?: (?:ge|le) \d+)*", line):
        return "prefix_lists"
    if bgp and re.fullmatch(r"neighbor \S+ prefix-list \S+ (?:in|out)", line):
        return "prefix_lists"
    if context.startswith("route-map ") and re.fullmatch(r"match ip address prefix-list \S+(?: \S+)*", line):
        return "prefix_lists"
    return None


def parse_extended(text, domain):
    result = {}
    for number, context, line in contextual_lines(text):
        if supported_setting(context, line) != domain:
            continue
        parts = line.split()
        if domain == "interfaces":
            if line.startswith("ip address "):
                ipaddress.IPv4Interface(f"{parts[2]}/{parts[3]}")
                key, value = "address", " ".join(parts[2:])
            elif line == "no ip address":
                key, value = "address", None
            elif "shutdown" in parts:
                key, value = "admin_state", "shutdown" if line == "shutdown" else "no shutdown"
            else:
                vlan = int(parts[2])
                if not 1 <= vlan <= 4094:
                    raise ValueError("Invalid VLAN ID.")
                key, value = "encapsulation", " ".join(parts[1:])
        elif domain == "bfd":
            if line.startswith("bfd interval "):
                values = [int(parts[i]) for i in (2, 4, 6)]
                if min(values) <= 0 or values[2] > 255:
                    raise ValueError("Invalid BFD numeric value; exact platform ranges are not validated.")
                key, value = "timers", dict(zip(("tx_ms", "min_rx_ms", "multiplier"), values))
            elif line in {"bfd echo", "no bfd echo"}:
                key, value = "echo", line == "bfd echo"
            else:
                peer = parts[2] if parts[0] == "no" else parts[1]
                key, value = f"neighbor {peer} registration", parts[0] != "no"
        else:
            if line.startswith("ip prefix-list "):
                network = ipaddress.IPv4Network(parts[6], strict=True)
                bounds = parts[7:]
                if len(set(bounds[::2])) != len(bounds[::2]):
                    raise ValueError("Duplicate prefix length qualifier.")
                qualifiers = dict(zip(bounds[::2], map(int, bounds[1::2])))
                ge, le = qualifiers.get("ge"), qualifiers.get("le")
                if (ge is not None and not network.prefixlen < ge <= 32) or (
                        le is not None and not network.prefixlen <= le <= 32) or (
                        ge is not None and le is not None and ge > le):
                    raise ValueError("Invalid prefix length qualifiers.")
                context = f"prefix-list {parts[2]}"
                key, value = f"seq {parts[4]}", " ".join(parts[5:])
            elif line.startswith("neighbor "):
                key, value = f"neighbor {parts[1]} {parts[-1]}", parts[3]
            else:
                key, value = "prefix-list match", parts[4:]
        full_key = f"{context} | {key}"
        if full_key in result:
            raise ValueError("Duplicate supported setting; requires manual inspection.")
        result[full_key] = {"value": value, "line": number}
    return result


def parse_settings(text, domain):
    """Use full block paths to distinguish peer AS and per-interface VRRP groups."""
    if domain not in DOMAINS:
        raise ValueError("Unsupported domain.")
    if domain in {"interfaces", "bfd", "prefix_lists"}:
        return parse_extended(text, domain)
    context = "global"
    result = {}
    pattern = (r"neighbor (\S+) remote-as (\d+)" if domain == "bgp"
               else r"vrrp (\d+) priority (\d+)")
    for number, line in enumerate(text.splitlines(), 1):
        if line and not line[0].isspace() and not line.startswith("!"):
            context = line.strip()
        match = re.fullmatch(pattern, line.strip())
        if match:
            if (domain == "bgp" and not context.startswith("router bgp ")) or (
                    domain == "vrrp" and not context.startswith("interface ")):
                raise ValueError("Supported setting outside its expected block.")
            key = f"{context} | {match[1]}"
            if key in result:
                raise ValueError("Duplicate supported setting; requires manual inspection.")
            result[key] = {"value": int(match[2]), "line": number}
    return result


def diff_proposal(state):
    if state["baseline"] is None:
        return {"error": "baseline_missing", "retryable": False}
    before, after = state["baseline"].splitlines(), state["proposal"].splitlines()
    diff = list(difflib.unified_diff(before, after, fromfile="approved", tofile="proposed", lineterm=""))
    domains, unsupported = set(), []
    snapshots = (list(contextual_lines(state["baseline"])), list(contextual_lines(state["proposal"])))
    for tag, a, b, c, d in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        for entries in (snapshots[0][a:b], snapshots[1][c:d]):
            for _, context, line in entries:
                domain = supported_setting(context, line)
                if domain:
                    domains.add(domain)
                elif line:
                    unsupported.append(line)
    state.update(diff=diff, required_domains=sorted(domains), unsupported=unsupported)
    if state.get("multi_agent") and not diff:
        state.setdefault("specialist_evidence", {})["comparison:L1"] = {
            "source": "comparison", "line": 1, "document_type": "computed_comparison",
            "text": "Completed baseline/proposal line comparison: no differences. This is not a live-state validation.",
            "baseline_sha256": state["baseline_hash"], "proposal_sha256": state["proposal_hash"]}
    return {"diff": diff, "required_checks": sorted(domains), "unsupported_changes": unsupported,
            "limit": LIMIT}


def make_report(state, summary, finalizing=False):
    missing = set(state.get("required_domains", [])) - set(state["checks"])
    missing_design = set(state.get("required_domains", [])) - set(state["design_read"])
    if not state.get("baseline_read") or "diff" not in state or missing or missing_design:
        return {"error": "prerequisites_incomplete", "missing_checks": sorted(missing),
                "missing_design": sorted(missing_design), "retryable": True}
    if state["unsupported"] and not state["unsupported_acknowledged"]:
        state.update(status="needs_human", human_kind="unsupported",
                     pending_question="Some changed lines cannot be validated. Acknowledge an explicitly limited review or cancel; approval cannot validate those lines.")
        return {"status": "needs_human", "unsupported": state["unsupported"]}
    if state.get("multi_agent") and not finalizing:
        state.update(configuration_summary=summary, agent_role="impact", role_reset=True)
        return {"status": "handoff", "to": "impact"}
    evidence = {"baseline": {"file": f"{state['router']}.txt", "sha256": state["baseline_hash"]},
                "design": {"file": "low-level-design.txt", "sha256": state["design_hash"],
                           "sections": state["design_read"]}}
    report = {"run_id": state.get("run_id"), "router": state["router"], "request": state["request"],
              "summary": summary, "checks": state["checks"], "diff": state["diff"],
              "unsupported_changes": state["unsupported"], "evidence": evidence,
              "proposal_sha256": state["proposal_hash"], "scope": LIMIT,
              "disposition": "limited_review" if state["unsupported"] else "baseline_review",
              "fault_injection": state["fault"], "human_clarifications": state["human_history"],
              "unresolved_questions": state.get("review_unknowns", []),
              "deployment_authorization": "not_granted_by_this_review"}
    if state.get("bgp_facts"):
        report["bgp_facts"] = state["bgp_facts"]
    state.update(report=report, report_hash=fingerprint(report), status="awaiting_approval",
                 human_kind="approval", pending_question="Review the exact report. Approve saving, reject it, or request a revision.")
    if state.get("bundle_devices"):
        from change_bundle import report_devices
        report.update(target_routers=state["target_routers"], devices=report_devices(state),
                      bundle_sha256=state["bundle_hash"], change_document=state["change_document"])
        report.pop("router")
        report.pop("proposal_sha256")
        report["evidence"].pop("baseline")
        state["report_hash"] = fingerprint(report)
    return {"status": "awaiting_approval", "report_hash": state["report_hash"]}


def dispatch(state, name, args):
    """The model cannot read arbitrary files or execute shell commands."""
    if state.get("bundle_devices") and name in {"inspect_proposal", "read_baseline", "compare_configs", "validate_settings"}:
        if state.get("agent_role", "configuration") != "configuration":
            return {"error": "tool_not_allowed_for_role", "retryable": False}
        from change_bundle import inspect_bundle
        return inspect_bundle(state, name, args)
    from corpus_tools import NAMES, dispatch_corpus
    if name in NAMES:
        return dispatch_corpus(state, name, args)
    if state.get("multi_agent") and state.get("agent_role", "configuration") != "configuration" and name != "ask_human" and not (
            state.get("agent_role") == "impact" and name == "read_design"):
        from specialists import specialist_dispatch
        return specialist_dispatch(state, name, args)
    shapes = {"read_baseline": set(), "inspect_proposal": set(), "compare_configs": set(), "validate_settings": {"domain"},
              "read_design": {"section"}, "ask_human": {"question"}, "submit_review": {"summary"}}
    if name == "ask_human" and isinstance(args, dict) and set(args) == {"question", "topic"}:
        from review_policy import TOPICS, human_question
        if not isinstance(args["topic"], str) or args["topic"] not in TOPICS or not isinstance(args["question"], str) or not 0 < len(args["question"].strip()) <= 3000:
            return {"error": "invalid_question", "retryable": False}
        return human_question(state, args["question"], args["topic"])
    if name not in shapes or not isinstance(args, dict) or set(args) != shapes[name]:
        return {"error": "invalid_tool_or_arguments", "retryable": False}
    if any(not isinstance(value, str) or not value.strip() or len(value) > 8000 for value in args.values()):
        return {"error": "invalid_argument_value", "retryable": False}
    if name == "read_baseline":
        if state["baseline"] is None:
            return {"error": "baseline_missing", "retryable": False}
        state["baseline_read"] = True
        return {"file": f"{state['router']}.txt", "text": state["baseline"], "sha256": state["baseline_hash"]}
    if name == "compare_configs":
        if state["fault"] == "permanent_compare" or (state["fault"] == "transient_compare"
                and state["tool_attempts"].get(name, 0) == 1):
            return {"error": "simulated_compare_unavailable", "retryable": True,
                    "note": "Explicit demo failure injection; not a real service outage."}
        return diff_proposal(state)
    if name == "inspect_proposal":
        if state["baseline"] is None:
            return {"error": "baseline_missing", "retryable": False}
        if state["fault"] == "permanent_compare" or (state["fault"] == "transient_compare" and
                state["tool_attempts"].get(name, 0) == 1):
            return {"error": "simulated_compare_unavailable", "retryable": True}
        state["baseline_read"] = True
        comparison = diff_proposal(state)
        if state["unsupported"] and not state["unsupported_acknowledged"]:
            state.update(status="needs_human", human_kind="unsupported",
                         pending_question="Unsupported changes require acknowledgment of a limited review, or cancellation.")
            return comparison
        design = {}
        for domain in state["required_domains"]:
            result = dispatch(state, "validate_settings", {"domain": domain})
            if "error" in result:
                return result
            design[domain] = dispatch(state, "read_design", {"section": domain})
            if "error" in design[domain]:
                return design[domain]
        return {**comparison, "checks": state["checks"], "design": design,
                "baseline_sha256": state["baseline_hash"],
                "note": "Bundled baseline inspection, diff, required checks and design lookup. No deployment judgment. Submit findings when sufficient."}
    if name == "validate_settings":
        domain = args["domain"]
        if domain not in DOMAINS:
            return {"error": "unsupported_domain", "retryable": False}
        if "diff" not in state:
            return {"error": "compare_first", "retryable": True}
        old, new = parse_settings(state["baseline"], domain), parse_settings(state["proposal"], domain)
        findings = [{"setting": key, "baseline": old.get(key), "proposal": new.get(key),
                     "finding": "departure_from_baseline"}
                    for key in sorted(old.keys() | new.keys())
                    if (old.get(key) or {}).get("value") != (new.get(key) or {}).get("value")]
        if domain == "prefix_lists":
            defined = {line.split()[2] for line in state["proposal"].splitlines()
                       if line.startswith("ip prefix-list ")}
            for number, context, line in contextual_lines(state["proposal"]):
                if supported_setting(context, line) != "prefix_lists" or line.startswith("ip prefix-list "):
                    continue
                names = [line.split()[3]] if line.startswith("neighbor ") else line.split()[4:]
                for reference in names:
                    if reference not in defined:
                        findings.append({"setting": f"{context} | {line}", "line": number,
                                         "finding": "undefined_prefix_list_reference", "reference": reference})
        state["checks"][domain] = findings
        if state.get("multi_agent"):
            from specialists import remember
            for finding in findings:
                for source in ("baseline", "proposal"):
                    item = finding.get(source)
                    if isinstance(item, dict) and item.get("line"):
                        remember(state, source, item["line"], state[source].splitlines()[item["line"] - 1])
        facts = None
        if domain == "bgp":
            from bgp_facts import collect_bgp_facts
            facts = collect_bgp_facts(state)
        return {"domain": domain, "findings": findings, "limit": LIMIT,
                **({"bgp_facts": facts} if facts is not None else {}),
                "note": "Missing explicit commands mean absence from this text, not a verified platform default. Prefix policy matching and end-to-end reachability are not evaluated."}
    if name == "read_design":
        section = args["section"]
        titles = {"bgp": "2. BFD and BGP", "vrrp": "3. VRRP",
                  "interfaces": "1. Addressing and physical links",
                  "bfd": "2. BFD and BGP", "prefix_lists": "2. BFD and BGP"}
        if section not in titles:
            return {"error": "unsupported_section", "retryable": False}
        text = state["design"] or ""
        lines = text.splitlines()
        start = next((i for i, line in enumerate(lines) if line == titles[section]), None)
        if start is None:
            return {"error": "design_section_missing", "retryable": False}
        end = next((i for i in range(start + 1, len(lines)) if re.match(r"^\d+\. ", lines[i])), len(lines))
        state["design_read"] = sorted(set(state["design_read"]) | {section})
        result = {"file": "low-level-design.txt", "section": titles[section],
                  "line_start": start + 1, "line_end": end, "text": "\n".join(lines[start:end]).strip()}
        state.setdefault("design_evidence", {})[section] = result
        if state.get("multi_agent"):
            from specialists import remember
            for index in range(start, end):
                remember(state, "low-level-design.txt", index + 1, lines[index])
        return result
    if name == "ask_human":
        from review_policy import human_question
        return human_question(state, args["question"])
    return make_report(state, args["summary"])


def schema(name, description, properties=None):
    properties = properties or {}
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(properties), "additionalProperties": False}}}


TOOLS = [schema("read_baseline", "Inspect the approved router baseline, with source and hash."),
         schema("inspect_proposal", "Preferred first tool: bundled baseline inspection, diff, required supported checks and relevant design evidence. Unknown settings still need human acknowledgment."),
         schema("compare_configs", "Find actual proposal differences and determine needed validation domains; retry once if a transient error is returned."),
         schema("validate_settings", "Compare explicit supported settings with baseline: BGP neighbor AS, VRRP priority, interface IPv4/admin state/dot1Q, BFD timers/echo/neighbor registration, or IPv4 prefix-list entries/references. Not complete syntax or impact validation.", {"domain": {"type": "string", "enum": list(DOMAINS)}}),
         schema("read_design", "Read local design evidence for a domain, with source lines.", {"section": {"type": "string", "enum": list(DOMAINS)}}),
         schema("ask_human", "Classify the question. Conditional mode records operational/documentary gaps as unknowns and limits intent clarification to one answered question. Never grants deployment approval.", {"question": {"type": "string"}, "topic": {"type": "string", "enum": ["intent", "authorization", "documentary_evidence", "operational_validation", "other"]}}),
         schema("submit_review", "Submit an evidence-backed summary for human review. Required checks enforced; never saves the final artifact.", {"summary": {"type": "string"}})]
