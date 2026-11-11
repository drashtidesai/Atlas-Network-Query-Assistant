"""Offline graph/control tests use explicit scripted models, not real agent reasoning."""

import copy
import json
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, messages_to_dict
from agent_context import working_messages, MAX_PACKET_CHARS
from review_graph import exhausted_limits, safe_error_details, provider_factory

from demo_cases import demo_proposal
from review_graph import ReviewService
from review_tools import create_state, dispatch, parse_settings
from corpus_tools import snapshot_corpus
from review_policy import finishing_tool, required_action
from change_bundle import bundle_state, uploaded_proposals
from bgp_facts import collect_bgp_facts, required_bgp_evidence


class ScriptedModel:
    def __init__(self, calls):
        self.calls = list(calls)
        self.count = 0

    def invoke(self, messages, **kwargs):
        if not self.calls:
            raise AssertionError("Scripted model exhausted; unexpected model call.")
        item = self.calls.pop(0)
        self.count += 1
        if isinstance(item, Exception):
            raise item
        name, args = item
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"t{self.count}"}],
                         usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})


def script(domains=("bgp",), retry=False):
    calls = [("read_baseline", {}), ("compare_configs", {})]
    if retry:
        calls += [("compare_configs", {})]
    for domain in domains:
        calls += [("validate_settings", {"domain": domain}), ("read_design", {"section": domain})]
    return calls + [("submit_review", {"summary": "Review based on observed tool findings; human review required."})]


class ReviewTests(unittest.TestCase):
    def test_59_verified_local_as_context_is_attached_transparently(self):
        state = self.bgp_state()
        state.update(agent_role="impact", specialist_queries=["find_references", "lookup_topology"])
        local = set(state["bgp_facts"]["LON-R01"]["local_as_evidence_ids"])
        cited = sorted(required_bgp_evidence(state) - local)
        result = dispatch(state, "submit_impact", {"assessment": "Remote-AS mismatch; local AS unchanged.",
                                                  "unknowns": "Live state unvalidated.", "evidence_ids": cited})
        self.assertEqual(result["status"], "handoff")
        self.assertEqual(set(state["impact_assessment"]["controller_added_context_ids"]), local)
        self.assertTrue(required_bgp_evidence(state) <= set(state["impact_assessment"]["evidence_ids"]))

    def test_60_missing_changed_neighbor_citation_is_still_rejected(self):
        state = self.bgp_state()
        state.update(agent_role="impact", specialist_queries=["find_references", "lookup_topology"])
        ids = required_bgp_evidence(state) - {"proposal:L158"}
        result = dispatch(state, "submit_impact", {"assessment": "Draft", "unknowns": "Live state unknown", "evidence_ids": sorted(ids)})
        self.assertEqual(result["error"], "bgp_configuration_evidence_required")
        self.assertEqual(result["missing_ids"], ["proposal:L158"])
        self.assertNotIn("impact_assessment", state)

    def test_55_submission_returns_exact_missing_ids(self):
        state = create_state("DXB-R01", demo_proposal("DXB-R01", "combined"), "Review")
        state.update(multi_agent=True, agent_role="configuration")
        dispatch(state, "inspect_proposal", {})
        self.assertIn("proposal:L104", state["specialist_evidence"])
        state["agent_role"] = "impact"
        result = dispatch(state, "submit_impact", {"assessment": "Draft", "unknowns": "Live state unknown", "evidence_ids": ["missing-file.txt:L4"]})
        self.assertEqual(result["missing_ids"], ["missing-file.txt:L4"])
        self.assertEqual(result["error"], "unknown_or_missing_evidence")

    def test_56_evidence_recovers_snapshot_lines_and_unique_aliases_only(self):
        state = create_state("DXB-R01", demo_proposal("DXB-R01", "combined"), "Review")
        state.update(multi_agent=True, agent_role="plan", corpus_snapshots={
            "corpus/synthetic-logs/example.txt": {"text": "simulated event\nsecond line", "document_type": "synthetic_log",
                                                  "sha256": "snapshot-hash", "evidence_is_simulated": True}})
        result = dispatch(state, "get_evidence", {"evidence_ids": ["proposal:L104", "example.txt:L1", "baseline:L99999", "../../secret.txt:L1"]})
        self.assertEqual([item["id"] for item in result["matches"]], ["proposal:L104", "corpus/synthetic-logs/example.txt:L1"])
        self.assertTrue(result["matches"][1]["evidence_is_simulated"])
        self.assertEqual(result["matches"][1]["text"], "simulated event")
        self.assertEqual(result["missing_ids"], ["baseline:L99999", "../../secret.txt:L1"])
        state["corpus_snapshots"]["corpus/design/example.txt"] = state["corpus_snapshots"]["corpus/synthetic-logs/example.txt"]
        result = dispatch(state, "get_evidence", {"evidence_ids": ["example.txt:L2"]})
        self.assertEqual(result["missing_ids"], ["example.txt:L2"])

    def test_57_combined_submission_recovers_missing_line_then_finishes(self):
        proposal = demo_proposal("DXB-R01", "combined")
        reference = create_state("DXB-R01", proposal, "Review")
        reference.update(multi_agent=True, corpus_snapshots=snapshot_corpus())
        dispatch(reference, "inspect_proposal", {})
        ids = sorted(required_bgp_evidence(reference)) + ["proposal:L103"]
        args = {"assessment": "Combined BGP and VRRP findings.", "unknowns": "Live behavior unverified.", "evidence_ids": ids}
        run_id = self.service.create("DXB-R01", proposal, "Review combined changes", multi_agent=True)
        models = {
            "configuration": ScriptedModel([("inspect_proposal", {}), ("submit_review", {"summary": "Two changed domains."})]),
            "impact": ScriptedModel([("find_references", {"query": "neighbor"}), ("lookup_topology", {"query": "DXB-R01"}),
                                      ("submit_impact", args), ("get_evidence", {"evidence_ids": ["baseline:L1"]}), ("submit_impact", args)]),
            "plan": ScriptedModel([("submit_plan", {"summary": "Combined conditional review.", "prechecks": "Confirm both changes.",
                                                     "verification": "Engineer validates both domains.", "rollback": "Restore changed neighbor and priority if authorized.",
                                                     "stop_conditions": "Evidence or authorization missing.", "evidence_ids": ids})])}
        result = self.service.advance(run_id, factory=lambda role: models[role])
        self.assertEqual(result["status"], "awaiting_approval")
        self.assertEqual(set(result["report"]["checks"]), {"bgp", "vrrp"})
        self.assertNotIn("evidence_recovery_ids", result)
        recovery = next(item for item in result["events"] if item["kind"] == "tool" and item["tool"] == "get_evidence")
        self.assertEqual(recovery["result"]["recovery_requested_ids"], ["proposal:L103"])
        self.assertEqual(recovery["result"]["model_requested_ids"], ["baseline:L1"])
        self.assertIsNone(self.service.store.artifact(run_id))

    def test_58_changing_submission_prose_cannot_reset_failure_bound(self):
        run_id = self.service.create("DXB-R01", demo_proposal("DXB-R01", "prefix_lists"), "Review", multi_agent=True)
        impact = [("find_references", {"query": "LOCAL-SITE"}), ("lookup_topology", {"query": "DXB-R01"})]
        for number in range(3):
            identifier = f"nonexistent-{number}.txt:L1"
            impact.append(("submit_impact", {"assessment": f"Different draft {number}", "unknowns": "Evidence missing", "evidence_ids": [identifier]}))
            if number < 2:
                impact.append(("get_evidence", {"evidence_ids": [identifier]}))
        models = {"configuration": ScriptedModel([("inspect_proposal", {}), ("submit_review", {"summary": "Changed prefix."})]),
                  "impact": ScriptedModel(impact)}
        result = self.service.advance(run_id, factory=lambda role: models[role])
        self.assertEqual(result["status"], "needs_human")
        self.assertEqual(result["human_kind"], "blocked")
        self.assertEqual(result["submission_failures"]["impact:submit_impact"], 3)
        self.assertLess(result["steps"], 18)
        self.assertNotIn("report", result)
        self.assertIsNone(self.service.store.artifact(run_id))

    def bgp_state(self):
        state = create_state("LON-R01", demo_proposal("LON-R01", "bgp"), "Review neighbor AS change")
        state.update(multi_agent=True, agent_role="configuration", corpus_snapshots=snapshot_corpus(), topology_snapshots={})
        dispatch(state, "inspect_proposal", {})
        return state

    def test_51_bgp_local_remote_and_peer_as_are_distinct(self):
        state = self.bgp_state()
        fact = state["bgp_facts"]["LON-R01"]
        self.assertEqual(fact["baseline_local_as"], [65020])
        self.assertEqual(fact["proposed_local_as"], [65020])
        self.assertFalse(fact["local_as_changed"])
        change = fact["neighbor_changes"][0]
        self.assertEqual(change["neighbor"], "10.0.0.1")
        self.assertEqual(change["baseline_remote_as"], 65010)
        self.assertEqual(change["proposed_remote_as"], 65030)
        self.assertEqual(change["documented_peers"][0]["local_as_values"], [65010])
        self.assertFalse(change["documented_peers"][0]["proposed_remote_as_matches"])
        self.assertTrue(required_bgp_evidence(state) <= set(state["specialist_evidence"]))

    def test_52_uploaded_peer_proposal_supersedes_corpus_intent(self):
        peer = demo_proposal("DXB-R01", "unchanged").replace("router bgp 65010", "router bgp 65030")
        state = bundle_state({"DXB-R01": peer, "LON-R01": demo_proposal("LON-R01", "bgp")}, "Review")
        state["corpus_snapshots"] = snapshot_corpus()
        owner = state["bundle_devices"]["LON-R01"]
        dispatch(owner, "compare_configs", {})
        dispatch(owner, "validate_settings", {"domain": "bgp"})
        fact = collect_bgp_facts(state, owner)
        peers = fact["neighbor_changes"][0]["documented_peers"]
        self.assertEqual(len(peers), 1)
        self.assertEqual(peers[0]["source"], "DXB-R01/proposal")
        self.assertEqual(peers[0]["local_as_values"], [65030])
        self.assertTrue(peers[0]["proposed_remote_as_matches"])

    def test_53_bgp_submission_requires_configuration_citations(self):
        state = self.bgp_state()
        state.update(agent_role="impact", specialist_queries=["find_references", "lookup_topology"])
        args = {"assessment": "Remote-AS change only; local AS unchanged.", "unknowns": "Live sessions unverified.",
                "evidence_ids": ["low-level-design.txt:L10"]}
        result = dispatch(state, "submit_impact", args)
        self.assertEqual(result["error"], "bgp_configuration_evidence_required")
        self.assertNotIn("impact_assessment", state)
        args["evidence_ids"] = sorted(required_bgp_evidence(state))
        self.assertEqual(dispatch(state, "submit_impact", args)["status"], "handoff")

    def test_54_bgp_facts_survive_working_context_and_final_report(self):
        state = self.bgp_state()
        packet = json.loads(working_messages(state, "Review")[1].content.split("\n", 1)[1])
        self.assertEqual(packet["bgp_facts"]["LON-R01"]["proposed_local_as"], [65020])
        state.update(agent_role="plan", impact_assessment={"assessment": "Remote-AS only", "unknowns": "Live state unknown"})
        result = dispatch(state, "submit_plan", {
            "summary": "Neighbor remote-AS changed; local AS unchanged.", "prechecks": "Confirm peer local AS.",
            "verification": "Engineer verifies named peer session.", "rollback": "Restore neighbor remote-as 65010 if authorized.",
            "stop_conditions": "Peer mismatch or missing authorization.", "evidence_ids": sorted(required_bgp_evidence(state))})
        self.assertEqual(result["status"], "awaiting_approval")
        self.assertFalse(state["report"]["bgp_facts"]["LON-R01"]["local_as_changed"])

    def test_47_unchanged_three_agent_review_finishes_without_dependency_search(self):
        run_id = self.service.create("DXB-R01", demo_proposal("DXB-R01", "unchanged"),
                                     "Report only actual changes.", multi_agent=True)
        ids = ["comparison:L1"]
        models = {
            "configuration": ScriptedModel([("inspect_proposal", {}), ("submit_review", {"summary": "No configuration differences."})]),
            "impact": ScriptedModel([("submit_impact", {"assessment": "No proposed changes; no change impact identified.",
                                                       "unknowns": "Live state unvalidated.", "evidence_ids": ids})]),
            "plan": ScriptedModel([("submit_plan", {"summary": "Proposal matches baseline.", "prechecks": "Review comparison.",
                                                     "verification": "No change verification required.", "rollback": "Not applicable; no change.",
                                                     "stop_conditions": "No deployment requested.", "evidence_ids": ids})])}
        result = self.service.advance(run_id, factory=lambda role: models[role])
        self.assertEqual(result["status"], "awaiting_approval")
        self.assertEqual(result["steps"], 4)
        self.assertEqual(result["diff"], [])
        self.assertEqual(result["report"]["checks"], {})
        self.assertEqual(result["specialist_evidence"]["comparison:L1"]["document_type"], "computed_comparison")
        self.assertIsNone(self.service.store.artifact(run_id))

    def test_48_impact_requires_missing_actions_after_discovery_allowance(self):
        state = create_state("DXB-R01", demo_proposal("DXB-R01", "prefix_lists"), "Review")
        state.update(multi_agent=True, efficient_review=True, agent_role="impact", role_turns={"impact": 2})
        self.assertIsNone(required_action(state))
        state["role_turns"]["impact"] = 3
        self.assertEqual(required_action(state), "find_references")
        state["specialist_queries"] = ["find_references"]
        self.assertEqual(required_action(state), "lookup_topology")
        with patch("langchain_openai.ChatOpenAI") as client:
            provider_factory("test-placeholder")("impact", {"lookup_topology"})
            self.assertEqual(client.return_value.bind_tools.call_args.kwargs["tool_choice"], "lookup_topology")

    def test_49_unsupported_continuation_progresses_after_inventory_calls(self):
        run_id = self.service.create("DXB-R01", demo_proposal("DXB-R01", "unsupported"),
                                     "Keep unsupported changes explicit.", multi_agent=True)
        cfg = ScriptedModel([("inspect_proposal", {}), ("inspect_proposal", {}),
                             ("submit_review", {"summary": "Description change unsupported; limited review only."})])
        baseline = create_state("DXB-R01", demo_proposal("DXB-R01", "unsupported"), "Review")["baseline"]
        line = next(i for i, text in enumerate(baseline.splitlines(), 1) if "description" in text)
        ids = [f"baseline:L{line}"]
        models = {"configuration": cfg,
                  "impact": ScriptedModel([("list_sources", {}), ("list_sources", {}),
                                            ("find_references", {"query": "description"}),
                                            ("lookup_topology", {"query": "DXB-R01"}),
                                            ("submit_impact", {"assessment": "Unsupported change; operational impact not certified.",
                                                               "unknowns": "Manual review required.", "evidence_ids": ids})]),
                  "plan": ScriptedModel([("submit_plan", {"summary": "Limited review.", "prechecks": "Manual review of unsupported description.",
                                                           "verification": "Engineer determines required checks.", "rollback": "No execution performed.",
                                                           "stop_conditions": "Unsupported change remains unvalidated.", "evidence_ids": ids})])}
        result = self.service.advance(run_id, factory=lambda role: models[role])
        self.assertEqual(result["human_kind"], "unsupported")
        result = self.service.advance(run_id, factory=lambda role: models[role], decision={
            "action": "acknowledge", "note": "Limited review only; do not certify unsupported changes."})
        self.assertEqual(result["status"], "awaiting_approval")
        self.assertEqual(result["report"]["disposition"], "limited_review")
        self.assertIsNone(self.service.store.artifact(run_id))

    def test_50_context_distinguishes_empty_diff_from_unrun_comparison(self):
        state = create_state("DXB-R01", demo_proposal("DXB-R01", "unchanged"), "Review")
        state.update(multi_agent=True)
        before = json.loads(working_messages(state, "Review")[1].content.split("\n", 1)[1])
        self.assertEqual(before["comparison_status"], "not_run")
        dispatch(state, "inspect_proposal", {})
        after = json.loads(working_messages(state, "Review")[1].content.split("\n", 1)[1])
        self.assertEqual(after["comparison_status"], "no_changes")
        self.assertTrue(after["baseline_read"])

    def test_46_ready_stage_requests_named_submission_from_provider(self):
        with patch("langchain_openai.ChatOpenAI") as client:
            factory = provider_factory("test-placeholder")
            for role, tool in (("configuration", "submit_review"), ("impact", "submit_impact"), ("plan", "submit_plan")):
                factory(role, {tool, "ask_human"})
                self.assertEqual(client.return_value.bind_tools.call_args.kwargs,
                                 {"tool_choice": tool, "parallel_tool_calls": False})
            factory("configuration")
            self.assertEqual(client.return_value.bind_tools.call_args.kwargs, {})

    def bundle(self, case="bfd"):
        return {router: demo_proposal(router, case) for router in ("DXB-R01", "LON-R01")}

    def test_42_compact_plan_context_has_no_duplicate_tool_text(self):
        state = bundle_state(self.bundle(), "Review both endpoints")
        state.update(multi_agent=True, agent_role="configuration", efficient_review=True)
        dispatch(state, "inspect_proposal", {})
        marker = "UNIQUE_SOURCE_TEXT_FOR_CONTEXT"
        identifier = "links.csv:L20"
        state["specialist_evidence"][identifier] = {"source": "links.csv", "line": 20, "text": marker, "sha256": "metadata" * 200}
        state.update(agent_role="plan", configuration_summary="PRIOR_CONFIGURATION_PROSE" * 100,
                     impact_assessment={"assessment": "Combined impact", "unknowns": "Live state unknown", "evidence_ids": [identifier]})
        state["messages"] = messages_to_dict([
            AIMessage(content="", tool_calls=[{"name": "get_evidence", "args": {"evidence_ids": [identifier]}, "id": "latest"}]),
            ToolMessage(content=json.dumps({"matches": [{"id": identifier, "text": marker}]}), tool_call_id="latest")])
        original = copy.deepcopy(state)
        context = working_messages(state, "Plan instructions")
        serialized = "\n".join(message.content for message in context)
        packet_text = context[1].content.split("\n", 1)[1]
        packet = json.loads(packet_text)
        self.assertEqual(serialized.count(marker), 1)
        self.assertNotIn("configuration_summary", packet)
        self.assertNotIn("device_proposals", packet)
        self.assertNotIn("diff", packet)
        self.assertLessEqual(len(packet_text), MAX_PACKET_CHARS)
        self.assertEqual(state, original)

    def test_43_compact_bundle_submits_only_after_required_evidence(self):
        state = bundle_state(self.bundle(), "Review")
        state.update(multi_agent=True, efficient_review=True, agent_role="configuration", role_turns={"configuration": 1})
        self.assertIsNone(finishing_tool(state))
        dispatch(state, "inspect_proposal", {})
        self.assertEqual(finishing_tool(state), "submit_review")
        state.update(agent_role="impact", role_turns={"impact": 3}, specialist_queries=["find_references"])
        self.assertIsNone(finishing_tool(state))
        state["specialist_queries"].append("lookup_topology")
        self.assertEqual(finishing_tool(state), "submit_impact")
        state.update(agent_role="plan", role_turns={"plan": 1})
        self.assertIsNone(finishing_tool(state))
        state["impact_assessment"] = {"assessment": "Both endpoints"}
        self.assertEqual(finishing_tool(state), "submit_plan")

    def test_44_compact_context_preserves_failed_lookup_and_missing_ids(self):
        state = create_state("DXB-R01", demo_proposal("DXB-R01", "bfd"), "Review")
        state["messages"] = messages_to_dict([
            AIMessage(content="", tool_calls=[{"name": "get_evidence", "args": {"evidence_ids": ["missing:L9"]}, "id": "missing"}]),
            ToolMessage(content=json.dumps({"error": "lookup_failed", "retryable": True,
                                            "missing_ids": ["missing:L9"]}), tool_call_id="missing")])
        packet = json.loads(working_messages(state, "Review")[1].content.split("\n", 1)[1])
        self.assertEqual(packet["last_tool_results"][0]["error"], "lookup_failed")
        self.assertEqual(packet["last_tool_results"][0]["missing_ids"], ["missing:L9"])
        self.assertTrue(packet["last_tool_results"][0]["retryable"])

    def test_45_oversized_handoff_excerpts_are_explicit_and_audit_intact(self):
        state = bundle_state(self.bundle(), "Conditional review")
        state.update(agent_role="plan", impact_assessment={"assessment": "long prose " * 3000, "evidence_ids": []},
                     messages=messages_to_dict([]))
        original = copy.deepcopy(state["impact_assessment"])
        packet_text = working_messages(state, "Plan")[1].content.split("\n", 1)[1]
        packet = json.loads(packet_text)
        self.assertIn("impact_assessment", packet["excerpted_fields"])
        self.assertGreater(packet["impact_assessment"]["omitted_characters"], 0)
        self.assertLessEqual(len(packet_text), MAX_PACKET_CHARS)
        self.assertEqual(state["impact_assessment"], original)

    def test_37_bundle_intake_and_document_hash(self):
        proposals = self.bundle()
        state = bundle_state(proposals, "Coordinated review", change_document="Both endpoints, planned only.")
        self.assertEqual(state["target_routers"], ["DXB-R01", "LON-R01"])
        different = bundle_state(proposals, "Coordinated review", change_document="Changed document")
        self.assertNotEqual(state["bundle_hash"], different["bundle_hash"])
        self.assertEqual(uploaded_proposals(list(proposals.values())), proposals)
        with self.assertRaises(ValueError):
            uploaded_proposals([proposals["DXB-R01"], proposals["DXB-R01"]])
        with self.assertRaises(ValueError):
            bundle_state({"DXB-R01": proposals["LON-R01"], "LON-R01": proposals["LON-R01"]}, "Review")
        with self.assertRaises(ValueError):
            bundle_state(proposals, "Review", change_document="x" * 20001)

    def test_38_bundle_inspection_covers_all_supported_domains(self):
        for domain in ("bgp", "vrrp", "interfaces", "bfd", "prefix_lists"):
            with self.subTest(domain=domain):
                state = bundle_state(self.bundle(domain), "Review")
                state.update(multi_agent=True, agent_role="configuration")
                result = dispatch(state, "inspect_proposal", {})
                self.assertNotIn("error", result)
                self.assertEqual({item["device"] for item in state["checks"][domain]}, {"DXB-R01", "LON-R01"})
                self.assertEqual(state["design_read"], [domain])
                self.assertTrue(state["baseline_read"])
                self.assertTrue(any(key.startswith("LON-R01/proposal:L") for key in result["evidence_ids"]))

    def test_39_bundle_evidence_is_device_scoped(self):
        state = bundle_state(self.bundle(), "Review")
        state.update(multi_agent=True, agent_role="impact")
        result = dispatch(state, "find_references", {"query": "bfd interval 1000"})
        self.assertEqual(len(result["matches"]), 2)
        ids = [item["id"] for item in result["matches"]]
        self.assertTrue(any(key.startswith("DXB-R01/proposal:") for key in ids))
        self.assertTrue(any(key.startswith("LON-R01/proposal:") for key in ids))
        self.assertEqual(len(dispatch(state, "get_evidence", {"evidence_ids": ids})["matches"]), 2)
        state["agent_role"] = "plan"
        self.assertEqual(len(dispatch(state, "inspect_change", {"query": ", ".join(ids)})["matches"]), 2)
        state["agent_role"] = "impact"
        missing = dispatch(state, "submit_impact", {"assessment": "Incomplete", "unknowns": "Live state unknown", "evidence_ids": ids[:1]})
        self.assertEqual(missing["error"], "bundle_device_evidence_required")

    def test_40_bundle_unsupported_and_missing_device_checks_block(self):
        proposals = self.bundle()
        proposals["LON-R01"] = demo_proposal("LON-R01", "unsupported")
        state = bundle_state(proposals, "Review")
        state.update(multi_agent=True, agent_role="configuration")
        dispatch(state, "inspect_proposal", {})
        self.assertEqual(state["status"], "needs_human")
        self.assertTrue(any(line.startswith("LON-R01:") for line in state["unsupported"]))
        result = dispatch(state, "submit_review", {"summary": "Cannot certify unsupported changes"})
        self.assertNotEqual(result.get("status"), "handoff")

    def test_41_bundle_three_agents_one_exact_approval(self):
        run_id = self.service.create("DXB-R01", "", "Review both endpoints; no deployment authorized.",
                                     multi_agent=True, proposals=self.bundle(), change_document="Coordinate both endpoints.")
        ids = [f"{router}/proposal:L90" for router in ("DXB-R01", "LON-R01")]
        models = {
            "configuration": ScriptedModel([("inspect_proposal", {}), ("submit_review", {"summary": "Both timer proposals reviewed."})]),
            "impact": ScriptedModel([("find_references", {"query": "bfd interval"}),
                                      ("lookup_topology", {"query": "WAN-A"}),
                                      ("submit_impact", {"assessment": "Both endpoints proposed at 1000ms; temporary partial state needs review.",
                                                         "unknowns": "Live negotiation unknown.", "evidence_ids": ids})]),
            "plan": ScriptedModel([("submit_plan", {"summary": "Coordinated offline review.", "prechecks": "Confirm exception and sequence.",
                                                     "verification": "Engineer verifies both endpoints.", "rollback": "Conditional restoration of both endpoints.",
                                                     "stop_conditions": "Partial implementation or failed checks; no deployment permission.", "evidence_ids": ids})])}
        result = self.service.advance(run_id, factory=lambda role: models[role])
        self.assertEqual(result["status"], "awaiting_approval")
        report = result["report"]
        self.assertEqual(set(report["devices"]), {"DXB-R01", "LON-R01"})
        self.assertEqual(report["bundle_sha256"], result["bundle_hash"])
        self.assertNotIn("router", report)
        self.assertIsNone(self.service.store.artifact(run_id))
        result = self.service.advance(run_id, decision=self.approval(run_id))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(self.service.store.artifact(run_id)["devices"], report["devices"])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = ReviewService(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def new(self, case="bgp", request="Review this proposal", fault="none"):
        return self.service.create("DXB-R01", demo_proposal("DXB-R01", case), request, fault)

    def advance(self, run_id, calls, decision=None):
        model = ScriptedModel(calls)
        result = self.service.advance(run_id, decision=decision, factory=lambda: model)
        return result

    def approval(self, run_id, action="approve"):
        data = self.service.snapshot(run_id)["data"]
        return {"action": action, "note": "Reviewed", "report_hash": data["report_hash"]}

    def test_01_unchanged_review_and_approval(self):
        run_id = self.new("unchanged")
        result = self.advance(run_id, script(()))
        self.assertEqual(result["status"], "awaiting_approval")
        self.assertEqual(result["diff"], [])
        self.assertIsNone(self.service.store.artifact(run_id))
        result = self.advance(run_id, [], self.approval(run_id))
        self.assertEqual(result["status"], "complete")
        self.assertIsNotNone(self.service.store.artifact(run_id))

    def test_02_bgp_findings(self):
        run_id = self.new()
        result = self.advance(run_id, script())
        finding = result["checks"]["bgp"][0]
        self.assertEqual(finding["baseline"]["value"], 65020)
        self.assertEqual(finding["proposal"]["value"], 65030)
        self.assertEqual(result["design_read"], ["bgp"])

    def test_03_vrrp_findings(self):
        result = self.advance(self.new("vrrp"), script(("vrrp",)))
        finding = result["checks"]["vrrp"][0]
        self.assertIn("GigabitEthernet0/0/1.110", finding["setting"])
        self.assertEqual(finding["proposal"]["value"], 90)

    def test_04_combined_checks(self):
        result = self.advance(self.new("combined"), script(("bgp", "vrrp")))
        self.assertEqual(set(result["checks"]), {"bgp", "vrrp"})
        self.assertEqual(result["status"], "awaiting_approval")

    def test_05_unsupported_forces_handoff(self):
        run_id = self.new("unsupported")
        result = self.advance(run_id, [("read_baseline", {}), ("compare_configs", {})])
        self.assertEqual(result["human_kind"], "unsupported")
        self.assertNotIn("report", result)
        result = self.advance(run_id, [("submit_review", {"summary": "Limited review; description change unsupported."})],
                              {"action": "acknowledge", "note": "Keep unsupported limitation visible"})
        self.assertEqual(result["report"]["disposition"], "limited_review")

    def test_06_clarification_resumes(self):
        run_id = self.new(request="")
        result = self.advance(run_id, [])
        self.assertEqual(result["human_kind"], "intake")
        self.assertEqual(result["steps"], 0)
        result = self.advance(run_id, script(), {"action": "reply", "note": "Review the changed peer AS"})
        self.assertEqual(result["status"], "awaiting_approval")

    def test_07_transient_tool_recovery(self):
        result = self.advance(self.new(fault="transient_compare"), script(retry=True))
        self.assertEqual(result["tool_attempts"]["compare_configs"], 2)
        self.assertEqual(result["status"], "awaiting_approval")

    def test_08_persistent_tool_error_stops(self):
        run_id = self.new(fault="permanent_compare")
        result = self.advance(run_id, [("read_baseline", {}), ("compare_configs", {}), ("compare_configs", {})])
        self.assertEqual(result["human_kind"], "blocked")
        self.assertEqual(result["tool_attempts"]["compare_configs"], 2)
        result = self.advance(run_id, [], {"action": "cancel", "note": "Investigate comparison failure"})
        self.assertEqual(result["status"], "unresolved")
        self.assertIsNone(self.service.store.artifact(run_id))

    def test_09_restart_reject_stale_and_duplicate_approval(self):
        run_id = self.new()
        self.advance(run_id, script())
        # New service/new SQLite connections represent a restarted caller.
        self.service = ReviewService(self.temp.name)
        state = self.service.snapshot(run_id)
        self.assertEqual(state["pending"][0]["kind"], "approval")
        with self.assertRaises(ValueError):
            self.advance(run_id, [], {"action": "approve", "report_hash": "stale"})
        correct = self.approval(run_id)
        self.advance(run_id, [], correct)
        with self.assertRaises(ValueError):
            self.advance(run_id, [], correct)
        self.assertEqual(len(self.service.store.list_runs()), 1)

    def test_10_rejection_and_revision(self):
        rejected = self.new()
        self.advance(rejected, script())
        result = self.advance(rejected, [], self.approval(rejected, "reject"))
        self.assertEqual(result["status"], "rejected")
        self.assertIsNone(self.service.store.artifact(rejected))
        revised = self.new()
        self.advance(revised, script())
        old = self.approval(revised)
        decision = {**old, "action": "revise", "note": "Make the baseline deviation clearer"}
        result = self.advance(revised, [("submit_review", {"summary": "Revised explanation of baseline deviation"})], decision)
        self.assertNotEqual(result["report_hash"], old["report_hash"])
        with self.assertRaises(ValueError):
            self.advance(revised, [], old)

    def test_11_unknown_tools_and_premature_submission(self):
        state = create_state("DXB-R01", demo_proposal("DXB-R01", "bgp"), "Review")
        self.assertIn("error", dispatch(state, "run_shell", {"command": "anything"}))
        self.assertIn("error", dispatch(state, "read_baseline", {"path": "outside"}))
        self.assertIn("error", dispatch(state, "submit_review", {"summary": "Looks safe"}))
        self.assertNotIn("report", state)

    def test_12_budget_and_model_error_recovery(self):
        run_id = self.new()
        result = self.advance(run_id, [TimeoutError()])
        self.assertEqual(result["human_kind"], "retry")
        result = self.advance(run_id, [TimeoutError()], {"action": "retry", "note": "One retry"})
        self.assertEqual(result["status"], "unresolved")
        exhausted = self.new()
        result = self.advance(exhausted, [("read_baseline", {})] * 10)
        self.assertEqual(result["status"], "budget_exhausted")

    def test_13_interface_change(self):
        result = self.advance(self.new("interfaces"), script(("interfaces",)))
        finding = result["checks"]["interfaces"][0]
        self.assertIn("GigabitEthernet0/0/0", finding["setting"])
        self.assertEqual(finding["proposal"]["value"], "shutdown")
        self.assertEqual(result["status"], "awaiting_approval")

    def test_14_bfd_change(self):
        result = self.advance(self.new("bfd"), script(("bfd",)))
        finding = result["checks"]["bfd"][0]
        self.assertEqual(finding["proposal"]["value"]["tx_ms"], 1000)
        self.assertEqual(result["required_domains"], ["bfd"])

    def test_15_prefix_action_change(self):
        result = self.advance(self.new("prefix_lists"), script(("prefix_lists",)))
        finding = result["checks"]["prefix_lists"][0]
        self.assertEqual(finding["baseline"]["value"], "permit 10.10.10.0/24")
        self.assertEqual(finding["proposal"]["value"], "deny 10.10.10.0/24")

    def test_16_prefix_deletion_and_missing_reference(self):
        proposal = demo_proposal("DXB-R01", "unchanged")
        proposal = "\n".join(line for line in proposal.splitlines()
                             if not line.startswith("ip prefix-list LOCAL-SITE "))
        state = create_state("DXB-R01", proposal, "Review")
        dispatch(state, "compare_configs", {})
        findings = dispatch(state, "validate_settings", {"domain": "prefix_lists"})["findings"]
        self.assertTrue(any(item["proposal"] is None for item in findings if "proposal" in item))
        self.assertTrue(any(item["finding"] == "undefined_prefix_list_reference" for item in findings))

    def test_17_extended_invalid_values_and_context(self):
        for text, domain in [
            ("interface Gi0\n ip address 999.0.0.1 255.255.255.0", "interfaces"),
            ("interface Gi0\n encapsulation dot1Q 5000", "interfaces"),
            ("interface Gi0\n bfd interval 0 min_rx 300 multiplier 3", "bfd"),
            ("ip prefix-list TEST seq 10 permit 10.0.0.0/24 ge 28 le 26", "prefix_lists")]:
            with self.subTest(domain=domain), self.assertRaises(ValueError):
                parse_settings(text, domain)
        state = create_state("DXB-R01", demo_proposal("DXB-R01", "unchanged") +
                             "\nline vty 16 20\n shutdown\n", "Review")
        result = dispatch(state, "compare_configs", {})
        self.assertIn("shutdown", result["unsupported_changes"])

    def test_18_bfd_registration_and_interface_deletion(self):
        proposal = demo_proposal("DXB-R01", "unchanged").replace(" neighbor 10.0.0.2 fall-over bfd\n", "")
        proposal = proposal.replace(" ip address 10.0.0.1 255.255.255.252\n", "")
        state = create_state("DXB-R01", proposal, "Review")
        self.assertEqual(dispatch(state, "compare_configs", {})["required_checks"], ["bfd", "interfaces"])
        for domain in ("bfd", "interfaces"):
            findings = dispatch(state, "validate_settings", {"domain": domain})["findings"]
            self.assertTrue(any(item["proposal"] is None for item in findings))

    def test_19_three_role_handoffs_and_approval(self):
        run_id = self.service.create("DXB-R01", demo_proposal("DXB-R01", "prefix_lists"),
                                     "Review policy change", multi_agent=True)
        evidence = ["baseline:L134"]
        configuration = ScriptedModel(script(("prefix_lists",)))
        impact = ScriptedModel([
            ("find_references", {"query": "LOCAL-SITE"}),
            ("lookup_topology", {"query": "DXB-R01"}),
            ("submit_impact", {"assessment": "Outbound policy dependency requires engineer verification.",
                               "unknowns": "Live advertisements unavailable.", "evidence_ids": evidence})])
        plan = ScriptedModel([("submit_plan", {"summary": "Offline proposal review; no execution.",
                             "prechecks": "Confirm intent.", "verification": "Engineer checks intended advertisements.",
                             "rollback": "Engineer reviews baseline restoration if authorized.",
                             "stop_conditions": "Unresolved intent or failed verification.", "evidence_ids": evidence})])
        models = {"configuration": configuration, "impact": impact, "plan": plan}
        result = self.service.advance(run_id, factory=lambda role: models[role])
        self.assertEqual(result["status"], "awaiting_approval")
        self.assertEqual(result["report"]["architecture"], "three_agent")
        self.assertEqual(result["report"]["plan_execution"], "not_executed")
        self.assertIsNone(self.service.store.artifact(run_id))
        self.assertEqual([item["to"] for item in result["events"] if item["kind"] == "agent_handoff"], ["impact", "plan"])
        old_hash = result["report_hash"]
        models["impact"] = ScriptedModel([("submit_impact", {
            "assessment": "Revised: effects limited to the named outbound neighbor; other paths unknown.",
            "unknowns": "Dependency search is incomplete.", "evidence_ids": evidence})])
        models["plan"] = ScriptedModel([("submit_plan", {
            "summary": "Revised neighbor-scoped summary.", "prechecks": "Confirm intent.",
            "verification": "Engineer verifies named neighbor advertisements.",
            "rollback": "Review baseline restoration if authorized.",
            "stop_conditions": "Unresolved evidence.", "evidence_ids": evidence})])
        result = self.service.advance(run_id, factory=lambda role: models[role], decision={
            "action": "revise", "note": "Qualify impact and plan together.", "report_hash": old_hash})
        self.assertEqual(result["status"], "awaiting_approval")
        self.assertIn("Revised:", result["report"]["impact_assessment"]["assessment"])
        self.assertNotEqual(result["report_hash"], old_hash)
        result = self.service.advance(run_id, decision=self.approval(run_id))
        self.assertEqual(result["status"], "complete")

    def test_36_late_clarification_refreshes_impact_before_report(self):
        run_id = self.service.create("DXB-R01", demo_proposal("DXB-R01", "prefix_lists"),
                                     "Review policy change", multi_agent=True)
        ids = ["baseline:L134"]
        models = {
            "configuration": ScriptedModel(script(("prefix_lists",))),
            "impact": ScriptedModel([
                ("find_references", {"query": "LOCAL-SITE"}),
                ("lookup_topology", {"query": "DXB-R01"}),
                ("submit_impact", {"assessment": "Original assessment", "unknowns": "Intent unknown", "evidence_ids": ids})]),
            "plan": ScriptedModel([("ask_human", {"question": "Confirm intent?", "topic": "intent"})])}
        result = self.service.advance(run_id, factory=lambda role: models[role])
        self.assertEqual(result["status"], "needs_human")
        models["impact"] = ScriptedModel([("submit_impact", {
            "assessment": "Human confirms accidental undeployed proposal.",
            "unknowns": "Live state unverified.", "evidence_ids": ids})])
        models["plan"] = ScriptedModel([("submit_plan", {
            "summary": "Accidental undeployed proposal.", "prechecks": "Correct proposal offline.",
            "verification": "Compare corrected proposal.", "rollback": "Not applicable: undeployed.",
            "stop_conditions": "No deployment authorization.", "evidence_ids": ids})])
        result = self.service.advance(run_id, factory=lambda role: models[role], decision={
            "action": "reply", "note": "Accidental and undeployed. No deployment authorized."})
        self.assertEqual(result["status"], "awaiting_approval")
        self.assertIn("Human confirms", result["report"]["impact_assessment"]["assessment"])
        self.assertEqual(result["report"]["impact_assessment"]["unknowns"], "Live state unverified.")
        self.assertIsNone(self.service.store.artifact(run_id))

    def test_20_specialist_permissions_and_missing_evidence(self):
        state = create_state("DXB-R01", demo_proposal("DXB-R01", "bfd"), "Review")
        state.update(multi_agent=True, agent_role="impact", topology_snapshots={})
        self.assertIn("error", dispatch(state, "validate_settings", {"domain": "bfd"}))
        self.assertIn("error", dispatch(state, "submit_impact", {"assessment": "Unknown", "unknowns": "Unknown", "evidence_ids": ["invented:L208"]}))
        self.assertNotIn("report", state)

    def test_21_topology_snapshot_and_no_early_handoff(self):
        run_id = self.service.create("DXB-R02", demo_proposal("DXB-R02", "vrrp"), "Review", multi_agent=True)
        state = self.service.snapshot(run_id)["data"]
        self.assertIn("DXB-R02", state["topology_snapshots"]["links.csv"])
        dispatch(state, "read_baseline", {})
        dispatch(state, "compare_configs", {})
        self.assertIn("error", dispatch(state, "submit_review", {"summary": "Premature"}))
        self.assertEqual(state["agent_role"], "configuration")

    def test_22_planner_resolves_configuration_design_and_topology_ids(self):
        state = create_state("DXB-R01", demo_proposal("DXB-R01", "prefix_lists"), "Review")
        state.update(multi_agent=True, agent_role="plan", specialist_evidence={
            "baseline:L134": {"source": "baseline", "line": 134, "text": "permit entry"},
            "low-level-design.txt:L11": {"source": "low-level-design.txt", "line": 11, "text": "design rule"},
            "addressing.csv:L5": {"source": "addressing.csv", "line": 5, "text": "topology row"}})
        ids = list(state["specialist_evidence"])
        result = dispatch(state, "get_evidence", {"evidence_ids": ids + ["baseline:L999"]})
        self.assertEqual(len(result["matches"]), 3)
        self.assertEqual(result["missing_ids"], ["baseline:L999"])
        recovered = dispatch(state, "inspect_change", {"query": ", ".join(ids)})
        self.assertEqual(len(recovered["matches"]), 3)

    def test_23_working_context_reuses_state_without_old_transcript(self):
        state = create_state("DXB-R01", demo_proposal("DXB-R01", "bfd"), "Review")
        state.update(multi_agent=True, max_steps=18, max_tokens=80000, steps=5,
                     human_history=[{"note": "Peer information unavailable; proceed conditionally."}],
                     specialist_evidence={"proposal:L90": {"text": "bfd interval 1000"}})
        state["messages"] = messages_to_dict([
            HumanMessage(content="OLD_VERBOSE_BLOCK" * 1000),
            AIMessage(content="", tool_calls=[{"name": "get_evidence", "args": {"evidence_ids": ["proposal:L90"]}, "id": "recent"}]),
            ToolMessage(content="Evidence retrieved", tool_call_id="recent")])
        original = copy.deepcopy(state["messages"])
        context = working_messages(state, "Planner instructions")
        text = " ".join(str(message.content) for message in context)
        self.assertNotIn("OLD_VERBOSE_BLOCK", text)
        self.assertIn("Peer information unavailable", text)
        self.assertIn("proposal:L90", text)
        self.assertEqual(len(context), 2)
        self.assertIsInstance(context[0], SystemMessage)
        self.assertIsInstance(context[1], HumanMessage)
        self.assertIn("Evidence retrieved", text)
        self.assertEqual(state["messages"], original)

    def test_35_evidence_context_is_bounded_and_audit_preserved(self):
        state = create_state("DXB-R01", demo_proposal("DXB-R01", "bfd"), "Review")
        state.update(steps=1, specialist_evidence={
            f"baseline:L{i}": {"text": str(i) + "x" * 1000}
            for i in range(40)})
        state["messages"] = messages_to_dict([HumanMessage(content="Review")])
        original = copy.deepcopy(state["specialist_evidence"])
        context = working_messages(state, "Review instructions")
        packet = json.loads(context[1].content.split("\n", 1)[1])
        self.assertEqual(len(packet["evidence_ids"]), 40)
        self.assertLess(len(packet["evidence"]), 40)
        self.assertLessEqual(sum(len(json.dumps(item, ensure_ascii=False))
                                 for item in packet["evidence"].values()), 12000)
        self.assertEqual(state["specialist_evidence"], original)

    def test_24_budget_reason_identifies_tokens_before_decision_limit(self):
        state = create_state("DXB-R01", demo_proposal("DXB-R01", "bfd"), "Review")
        state.update(steps=16, max_steps=18, max_tokens=80000,
                     usage={"input_tokens": 79379, "output_tokens": 1351})
        self.assertEqual(exhausted_limits(state), ["reported_tokens"])

    def test_25_api_diagnostics_exclude_exception_payload(self):
        error = RuntimeError("SECRET_KEY_AND_RESPONSE_BODY")
        error.status_code = 429
        error.code = "rate_limit_exceeded"
        error.body = {"credential": "SECRET_KEY_AND_RESPONSE_BODY"}
        details = safe_error_details(error)
        self.assertEqual(details["http_status"], 429)
        self.assertEqual(details["provider_code"], "rate_limit_exceeded")
        self.assertNotIn("SECRET_KEY", str(details))
        error.code = "Bearer SECRET_KEY"
        self.assertNotIn("provider_code", safe_error_details(error))

    def test_26_london_router_demos_use_site_specific_settings(self):
        for router in ("LON-R01", "LON-R02"):
            for domain in ("bgp", "vrrp", "bfd", "interfaces", "prefix_lists"):
                with self.subTest(router=router, domain=domain):
                    state = create_state(router, demo_proposal(router, domain), "Review")
                    result = dispatch(state, "compare_configs", {})
                    self.assertEqual(result["required_checks"], [domain])
                    findings = dispatch(state, "validate_settings", {"domain": domain})["findings"]
                    self.assertTrue(findings)
                    if domain == "bgp":
                        self.assertEqual(findings[0]["baseline"]["value"], 65010)
                    if domain == "vrrp":
                        self.assertIn(".120", findings[0]["setting"])
                        self.assertEqual(findings[0]["proposal"]["value"], 90)

    def test_27_corpus_inventory_peer_evidence_and_labels(self):
        state = create_state("DXB-R01", demo_proposal("DXB-R01", "bfd"), "Review")
        state["corpus_snapshots"] = snapshot_corpus()
        sources = dispatch(state, "list_sources", {})["sources"]
        self.assertEqual(len(sources), 27)
        self.assertFalse(any("evaluation" in item["source"] for item in sources))
        peer = "corpus/intended-configs/LON-R01.txt"
        document = state["corpus_snapshots"][peer]
        line = next(i for i, text in enumerate(document["text"].splitlines(), 1)
                    if "bfd interval" in text)
        result = dispatch(state, "read_source", {"source": peer, "line_start": line, "line_end": line})
        self.assertIn("300", result["matches"][0]["text"])
        self.assertEqual(result["matches"][0]["document_type"], "intended_configuration")
        self.assertIn(peer, state["corpus_accessed"])
        simulated = next(item for item in sources if item["document_type"] == "synthetic_log")
        result = dispatch(state, "read_source", {"source": simulated["source"], "line_start": 1, "line_end": 1})
        self.assertTrue(result["matches"][0]["evidence_is_simulated"])

    def test_28_corpus_read_boundaries_and_search_limit(self):
        state = create_state("DXB-R01", demo_proposal("DXB-R01", "bgp"), "Review")
        state["corpus_snapshots"] = snapshot_corpus()
        for args in [{"source": "../../.env", "line_start": 1, "line_end": 1},
                     {"source": "corpus/intended-configs/LON-R01.txt", "line_start": 1, "line_end": 41},
                     {"source": "corpus/intended-configs/LON-R01.txt", "line_start": True, "line_end": 2}]:
            self.assertIn("error", dispatch(state, "read_source", args))
        result = dispatch(state, "search_sources", {"query": "LON-R01 bfd"})
        self.assertTrue(result["matches"])
        self.assertLessEqual(len(result["matches"]), 8)
        self.assertTrue(all("document_type" in item for item in result["matches"]))

    def test_29_corpus_reads_use_snapshot_not_disk(self):
        state = create_state("DXB-R01", demo_proposal("DXB-R01", "bfd"), "Review")
        state["corpus_snapshots"] = {"corpus/design/example.txt": {
            "text": "Immutable saved content", "document_type": "design_document",
            "sha256": "saved-hash", "line_count": 1, "evidence_is_simulated": False}}
        result = dispatch(state, "read_source", {"source": "corpus/design/example.txt", "line_start": 1, "line_end": 1})
        self.assertEqual(result["matches"][0]["text"], "Immutable saved content")
        self.assertEqual(result["matches"][0]["sha256"], "saved-hash")

    def test_30_historical_review_cannot_silently_gain_new_corpus(self):
        state = create_state("DXB-R01", demo_proposal("DXB-R01", "bgp"), "Review")
        self.assertEqual(dispatch(state, "list_sources", {})["error"], "corpus_not_snapshotted_for_this_run")

    def test_31_bundled_inspection_gathers_required_checks(self):
        state = create_state("LON-R01", demo_proposal("LON-R01", "combined"), "Review")
        result = dispatch(state, "inspect_proposal", {})
        self.assertNotIn("error", result)
        self.assertTrue(state["baseline_read"])
        self.assertEqual(set(state["checks"]), {"bgp", "vrrp"})
        self.assertEqual(set(state["design_read"]), {"bgp", "vrrp"})
        self.assertNotIn("report", state)
        unsupported = create_state("DXB-R01", demo_proposal("DXB-R01", "unsupported"), "Review")
        dispatch(unsupported, "inspect_proposal", {})
        self.assertEqual(unsupported["human_kind"], "unsupported")

    def test_32_conditional_questions_record_unknowns_without_approval(self):
        state = create_state("DXB-R01", demo_proposal("DXB-R01", "bfd"), "Conditional review")
        state.update(efficient_review=True)
        result = dispatch(state, "ask_human", {"question": "Are live peer timers confirmed?", "topic": "operational_validation"})
        self.assertEqual(result["status"], "recorded_unknown")
        self.assertEqual(state["status"], "running")
        self.assertEqual(len(state["review_unknowns"]), 1)
        result = dispatch(state, "ask_human", {"question": "Is this intentional?", "topic": "intent"})
        self.assertEqual(result["status"], "needs_human")
        state["human_history"] = [{"kind": "clarification", "action": "reply", "note": "Intent unknown; conditional report only."}]
        state["status"] = "running"
        result = dispatch(state, "ask_human", {"question": "Another question?", "topic": "other"})
        self.assertEqual(result["status"], "recorded_unknown")
        dispatch(state, "inspect_proposal", {})
        dispatch(state, "submit_review", {"summary": "Conditional finding; unknowns remain."})
        self.assertEqual(state["status"], "awaiting_approval")
        self.assertEqual(len(state["report"]["unresolved_questions"]), 2)
        self.assertEqual(state["report"]["deployment_authorization"], "not_granted_by_this_review")

    def test_33_finish_guidance_requires_completed_evidence(self):
        state = create_state("DXB-R01", demo_proposal("DXB-R01", "bfd"), "Review")
        state.update(efficient_review=True, agent_role="configuration", role_turns={"configuration": 4})
        self.assertIsNone(finishing_tool(state))
        dispatch(state, "inspect_proposal", {})
        self.assertEqual(finishing_tool(state), "submit_review")
        state.update(agent_role="impact", role_turns={"impact": 4}, specialist_queries=["find_references"])
        self.assertIsNone(finishing_tool(state))
        state["specialist_queries"].append("lookup_topology")
        state["specialist_evidence"] = {"baseline:L90": {"text": "bfd"}}
        self.assertEqual(finishing_tool(state), "submit_impact")

    def test_34_fast_multi_agent_review_still_requires_human_save(self):
        run_id = self.service.create("DXB-R01", demo_proposal("DXB-R01", "bfd"), "Conditional review", multi_agent=True)
        ids = ["low-level-design.txt:L10"]
        models = {
            "configuration": ScriptedModel([("inspect_proposal", {}), ("submit_review", {"summary": "Changed BFD timers."})]),
            "impact": ScriptedModel([("find_references", {"query": "bfd"}),
                                       ("lookup_topology", {"query": "DXB-R01"}),
                                       ("submit_impact", {"assessment": "Documented peer dependency.", "unknowns": "Live timers unverified.", "evidence_ids": ids})]),
            "plan": ScriptedModel([("submit_plan", {"summary": "Conditional review only.", "prechecks": "Verify peer settings.",
                                     "verification": "Engineer confirms negotiated behavior.", "rollback": "Engineer reviews baseline restoration if authorized.",
                                     "stop_conditions": "No implementation until validation.", "evidence_ids": ids})])}
        result = self.service.advance(run_id, factory=lambda role: models[role])
        self.assertEqual(result["steps"], 6)
        self.assertEqual(result["status"], "awaiting_approval")
        self.assertIsNone(self.service.store.artifact(run_id))


if __name__ == "__main__":
    unittest.main(verbosity=2)
