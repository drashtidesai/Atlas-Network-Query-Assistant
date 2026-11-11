"""Deterministic multi-device intake and inspection; no provider calls."""
import re

from review_tools import create_state, dispatch, fingerprint


def uploaded_proposals(texts):
    proposals = {}
    for text in texts:
        matches = re.findall(r"^hostname (\S+)\s*$", text, re.M)
        if len(matches) != 1 or matches[0] in proposals:
            raise ValueError("Each file must have exactly one unique device hostname.")
        proposals[matches[0]] = text
    return proposals


def bundle_state(proposals, request, fault="none", change_document=""):
    if not isinstance(proposals, dict) or not 2 <= len(proposals) <= 4:
        raise ValueError("A bundle requires two to four supported router configurations.")
    if not isinstance(change_document, str) or len(change_document.encode()) > 20000:
        raise ValueError("Change document must be UTF-8 text of at most 20 KB.")
    if any(not isinstance(text, str) or len(re.findall(r"^hostname (\S+)\s*$", text, re.M)) != 1
           for text in proposals.values()):
        raise ValueError("Each proposal must have exactly one hostname.")
    devices = {router: create_state(router, text, request, fault)
               for router, text in sorted(proposals.items())}
    if any(device["baseline"] is None for device in devices.values()):
        raise ValueError("Every bundle device requires an available baseline.")
    first = next(iter(devices.values()))
    state = create_state(first["router"], first["proposal"], request, fault)
    state.update(bundle_devices=devices, target_routers=list(devices), compact_workflow=True,
                 change_document=change_document,
                 bundle_hash=fingerprint({"devices": {
                     router: {"baseline": item["baseline_hash"], "proposal": item["proposal_hash"]}
                     for router, item in devices.items()}, "change_document": change_document}))
    return state


def inspect_bundle(state, name, args):
    expected = {"read_baseline": set(), "compare_configs": set(),
                "inspect_proposal": set(), "validate_settings": {"domain"}}
    if not isinstance(args, dict) or set(args) != expected[name]:
        return {"error": "invalid_bundle_arguments", "retryable": False}
    results = {}
    for router, device in state["bundle_devices"].items():
        device["unsupported_acknowledged"] = state["unsupported_acknowledged"]
        device["tool_attempts"] = state["tool_attempts"]
        result = dispatch(device, name, args)
        if "error" in result:
            return {**result, "device": router}
        results[router] = {**result, "baseline_source": f"{router}/baseline",
                           "proposal_source": f"{router}/proposal"}
        if "bgp" in device["checks"]:
            from bgp_facts import collect_bgp_facts
            results[router]["bgp_facts"] = collect_bgp_facts(state, device)
        from specialists import remember
        for findings in device["checks"].values():
            for finding in findings:
                for source in ("baseline", "proposal"):
                    record = finding.get(source)
                    if isinstance(record, dict) and record.get("line"):
                        number = record["line"]
                        remember(state, f"{router}/{source}", number,
                                 device[source].splitlines()[number - 1])
    devices = state["bundle_devices"]
    if all("diff" in item for item in devices.values()):
        state["diff"] = [line for router, item in devices.items()
                         for line in ([f"# Device: {router}"] + item["diff"])]
        state["required_domains"] = sorted({domain for item in devices.values()
                                             for domain in item["required_domains"]})
        state["unsupported"] = [f"{router}: {line}" for router, item in devices.items()
                                for line in item["unsupported"]]
    state["baseline_read"] = all(item.get("baseline_read") for item in devices.values())
    # A domain is complete only when every device changing it was checked.
    for domain in state.get("required_domains", []):
        affected = [item for item in devices.values() if domain in item["required_domains"]]
        if all(domain in item["checks"] for item in affected):
            state["checks"][domain] = [{"device": router, **finding}
                for router, item in devices.items() for finding in item["checks"].get(domain, [])]
    if name == "inspect_proposal":
        if state.get("unsupported") and not state["unsupported_acknowledged"]:
            state.update(status="needs_human", human_kind="unsupported",
                         pending_question="Bundle contains unsupported changes. Acknowledge a limited review or cancel.")
        else:
            for domain in state.get("required_domains", []):
                dispatch(state, "read_design", {"section": domain})
    return {"devices": results, "bundle_sha256": state["bundle_hash"],
            "evidence_ids": [key for key in state.get("specialist_evidence", {})
                             if "/baseline:L" in key or "/proposal:L" in key],
            "note": "All device proposals are one coordinated proposed state, not observed running state."}


def report_devices(state):
    return {router: {"baseline_file": router + ".txt", "baseline_sha256": item["baseline_hash"],
                     "proposal_sha256": item["proposal_hash"], "checks": item["checks"],
                     "diff": item.get("diff", []), "unsupported_changes": item.get("unsupported", [])}
            for router, item in state["bundle_devices"].items()}
