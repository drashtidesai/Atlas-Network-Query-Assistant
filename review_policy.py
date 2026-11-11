"""Conditional offline-review policy; unknowns never become approvals."""
TOPICS = ("intent", "authorization", "documentary_evidence", "operational_validation", "other")


def no_changes(state):
    """An executed empty diff is a result, not missing input."""
    return state.get("baseline_read", False) and "diff" in state and not state["diff"] and not state.get("unsupported")


def human_question(state, question, topic="other"):
    answered = sum(item.get("action") == "reply" and item.get("kind") == "clarification"
                   for item in state.get("human_history", []))
    if state.get("efficient_review") and (topic in {"operational_validation", "documentary_evidence"} or answered >= 1):
        record = {"question": question, "topic": topic,
                  "resolution": "unverified; engineer confirmation required before implementation"}
        records = state.setdefault("review_unknowns", [])
        if record not in records:
            records.append(record)
        return {"status": "recorded_unknown", "unknown": record,
                "instruction": "Continue conditional offline review. Search available documents if relevant. Do not infer answers, intent, authorization, live support or deployment permission. Human approval remains required."}
    state.update(status="needs_human", human_kind="clarification", pending_question=question)
    return {"status": "needs_human", "question": question}


def finishing_tool(state):
    if not state.get("efficient_review"):
        return None
    role = state.get("agent_role", "configuration")
    turns = state.get("role_turns", {}).get(role, 0)
    if role == "configuration":
        ready = state.get("baseline_read") and "diff" in state and (
            set(state.get("required_domains", [])) <= set(state["checks"]) and
            set(state.get("required_domains", [])) <= set(state["design_read"]))
        return "submit_review" if ready and (no_changes(state) or state.get("compact_workflow") or turns >= 4) else None
    if role == "impact":
        if no_changes(state):
            return "submit_impact"
        ready = {"find_references", "lookup_topology"} <= set(state.get("specialist_queries", []))
        allowance = 3 if state.get("compact_workflow") else 4
        return "submit_impact" if ready and state.get("specialist_evidence") and turns >= allowance else None
    return "submit_plan" if state.get("impact_assessment") and (no_changes(state) or state.get("compact_workflow") or turns >= 2) else None


def required_action(state):
    """Reserve missing impact prerequisites after two free investigation turns.

    No fabricated lookup results or automatic report approval: the model must
    provide valid arguments and the tool still executes its ordinary checks.
    """
    if state.get("evidence_recovery_ids"):
        return "get_evidence"
    finish = finishing_tool(state)
    if finish:
        return finish
    if (state.get("efficient_review") and state.get("agent_role") == "impact" and
            state.get("role_turns", {}).get("impact", 0) >= 3):
        for tool in ("find_references", "lookup_topology"):
            if tool not in state.get("specialist_queries", []):
                return tool
    return None
