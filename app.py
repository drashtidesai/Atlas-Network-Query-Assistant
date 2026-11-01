"""Atlas Network Evidence Assistant. Launch: uv run streamlit run app.py"""

import json

import streamlit as st

from check_chunk_lengths import DEFAULT_EXPORT, read_chunks
from product_pipeline import generate_product, retrieve_product


st.set_page_config(page_title="Atlas | Network Evidence", page_icon="🌐", layout="wide")


@st.cache_data(show_spinner=False)
def load_export(modified_ns: int) -> list[dict]:
    return read_chunks(DEFAULT_EXPORT)


st.title("Atlas Network Evidence Assistant")
st.write("Explore network designs, configurations and failure scenarios with traceable sources.")
st.caption("Fictional training network · Synthetic logs · Answers are unverified drafts")

try:
    records = load_export(DEFAULT_EXPORT.stat().st_mtime_ns)
except (OSError, ValueError):
    st.error("Cannot load work/chunks.jsonl. Restore or rebuild the chunk export before starting.")
    st.stop()

with st.sidebar:
    st.header("Connection")
    mode = st.selectbox("Retrieval", ["Local keyword search", "Hybrid: semantic + keyword"])
    method = "hybrid" if mode.startswith("Hybrid") else "local"
    pinecone_key = st.text_input("Pinecone API key", type="password", disabled=method == "local")
    nebius_key = st.text_input("Nebius API key", type="password")
    st.caption("Keys stay in this browser session's app memory; the app does not write them to files. Use trusted local hosting.")
    with st.expander("Retrieval settings"):
        evidence_limit = st.slider("Maximum evidence chunks", 4, 20, 12)
    st.caption("Local search needs no API key. Hybrid search uses Pinecone quota. Each draft uses one paid Nebius call.")
    if st.button("Clear session"):
        st.session_state.clear()
        st.rerun()

examples = [
    "In synthetic FAIL-01, why did DXB-R01 remain the VRRP master after WAN-A silently failed?",
    "In synthetic FAIL-08, what happened to TACACS+, syslog, SNMP and NTP during the management service failures?",
    "In synthetic FAIL-99, what happened after WAN-A failed?",
]
choice = st.selectbox("Try a demo question", ["Write my own question"] + examples)
question = st.text_area("Your network question", value="" if choice == "Write my own question" else choice,
                        height=95, key=f"question_{choice}")
signature = (question.strip(), method, evidence_limit)

if st.button("Find evidence", type="primary", disabled=not question.strip()):
    st.session_state.pop("answer", None)
    st.session_state.pop("retrieved", None)
    try:
        with st.spinner("Searching sources and checking scenario coverage…"):
            result = retrieve_product(question, records, method, pinecone_key,
                                      evidence_limit=evidence_limit)
        st.session_state["retrieved"] = result
        st.session_state["signature"] = signature
    except Exception as error:
        st.error(f"Retrieval could not finish ({type(error).__name__}). Check the selected connection and retry. No answer was generated.")

result = st.session_state.get("retrieved")
if result and st.session_state.get("signature") != signature:
    st.info("The question or retrieval settings changed. Find evidence again to continue.")
    st.stop()
if not result:
    st.info("Choose a question, then find evidence. You can inspect sources before generating a draft.")
    st.stop()

package = result["package"]
ready = result["route"] == "evidence_review"
left, middle, right = st.columns(3)
left.metric("Evidence chunks", len(package["evidence"]))
middle.metric("Corpus chunks", len(records))
right.metric("Coverage route", result["route"])
if ready:
    st.success("Evidence retrieved. Review the sources below, then optionally generate a draft.")
    st.caption("Scenario coverage checks do not establish that every claim can be answered.")
else:
    st.warning(result["message"])
st.caption("Workflow: " + " → ".join(["START"] + result["trace"] + ["END"]))

answer_tab, evidence_tab, details_tab = st.tabs(["Answer", "Sources", "Coverage & export"])
with answer_tab:
    if st.button("Generate cited draft", disabled=not ready or not nebius_key.strip()):
        st.session_state.pop("answer", None)
        try:
            with st.spinner("Preparing a draft from the selected evidence…"):
                st.session_state["answer"] = generate_product(package, records, nebius_key)
        except Exception as error:
            st.error(f"Generation could not finish ({type(error).__name__}). A submitted request may have incurred usage.")
    if ready and not nebius_key.strip():
        st.caption("Enter a Nebius key in the sidebar to generate a draft.")
    answer = st.session_state.get("answer")
    if answer:
        if answer["issues"]:
            st.warning("Draft withheld because a local check flagged an issue.")
            for issue in answer["issues"]:
                st.write(issue)
        else:
            st.markdown(answer["draft"])
            st.caption("Unverified draft. Citation labels establish references, not factual correctness. Check each claim against Sources.")
            st.download_button("Download draft", answer["draft"], "atlas-draft.txt", "text/plain")
        with st.expander("Generation usage"):
            st.json(answer["usage"])
with evidence_tab:
    for item in package["evidence"]:
        with st.expander(f"[{item['citation_id']}] {item['source']} · {item['section']}"):
            st.caption("Source lines: " + ", ".join(item["source_line_ranges"]) + " · " + item["document_type"])
            st.text(item["text"])
            st.caption("Chunk ID: " + item["chunk_id"])
with details_tab:
    if package["scenario_coverage"]:
        st.dataframe([{"Scenario": row["scenario_id"], "Status": row["status"],
                       "Available logs": row["available_log_chunk_count"],
                       "Selected logs": row["selected_log_chunk_count"]}
                      for row in package["scenario_coverage"]], hide_index=True)
    else:
        st.caption("No explicit FAIL-xx scenario was requested; scenario coverage was not evaluated.")
    st.download_button("Download evidence package", json.dumps(package, indent=2, ensure_ascii=False),
                       "atlas-evidence.json", "application/json")
    st.caption("The package contains source text and provenance, never API keys.")
