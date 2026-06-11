"""FitScience demo UI: chat with the agent, upload new research PDFs."""
import tempfile
from pathlib import Path

import streamlit as st

from src.agent import ask, make_agent
from src.ingestion_pipeline import _slugify, ingest_paper
from src.viz import render_evidence_graph

st.set_page_config(page_title="FitScience", page_icon="💪", layout="wide")
st.title("💪 FitScience — Evidence-Based Lifting Assistant")
st.caption("Hybrid GraphRAG over a fitness knowledge graph in ArangoDB")


@st.cache_resource
def get_agent():
    return make_agent()


if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: PDF ingestion
with st.sidebar:
    st.header("📄 Add a research paper")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded and st.button("Ingest into knowledge graph"):
        with st.spinner("Extracting, chunking, embedding..."):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            result = ingest_paper(tmp_path, _slugify(Path(uploaded.name).stem))
            Path(tmp_path).unlink(missing_ok=True)
        msg = f"Ingested **{result['title']}** — {result['claims']} claims, {result['chunks']} chunks"
        if result.get("contradictions_found"):
            msg += f"\n\n⚠️ This paper **contradicts {result['contradictions_found']} existing claim(s)** — the graph was updated."
        st.success(msg)

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if query := st.chat_input("Ask a lifting question backed by science..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge graph..."):
            result = ask(get_agent(), query, thread_id="streamlit-session")
        meta = f"`plan: {result['retrieval_plan']}` · `contradictions: {len(result['contradictions'])}`"
        reply = f"{result['final_answer']}\n\n---\n{meta}"
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.last_result = result

# Evidence graph for the most recent answer. Rendered outside the chat flow on
# every rerun — a vis.js canvas inside a collapsed expander gets zero size, so a
# toggle (which re-renders at full size when switched on) is used instead.
if st.session_state.get("last_result"):
    st.divider()
    if st.toggle("🕸️ Show evidence graph for the last answer"):
        render_evidence_graph(st.session_state.last_result)
