import sys
import traceback
from pathlib import Path

_frontend_dir = str(Path(__file__).resolve().parent)
if _frontend_dir not in sys.path:
    sys.path.insert(0, _frontend_dir)

import streamlit as st
from backend import Backend

st.set_page_config(page_title="Local RAG", page_icon="📚", layout="wide")

for key, default in [
    ("backend", Backend()),
    ("nav", "Home"),
    ("messages", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

backend = st.session_state.backend

with st.sidebar:
    st.title("System Status")

    health = backend.health()
    if health.get("chroma"):
        st.success("Chroma \u2705")
    else:
        st.error("Chroma \u274c")

    if health.get("ollama"):
        st.success("Ollama \u2705")
    else:
        st.error("Ollama \u274c")

    st.divider()
    docs = backend.document_count()
    chunks = backend.chunk_count()
    st.metric("Documents", docs)
    st.metric("Chunks", chunks)
    st.divider()

    st.radio("Navigation", ["Home", "Ingestion", "Chat"], key="nav")

page = st.session_state.nav

if page == "Home":
    st.title("\U0001f4da Local RAG Document Assistant")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### \U0001f4c4 Ingestion")
        st.markdown("Index documents into the vector database.")
        if st.button("Start Ingestion", use_container_width=True, type="primary"):
            st.session_state.nav = "Ingestion"
            st.rerun()

    with col2:
        st.markdown("### \U0001f4ac Chat")
        st.markdown("Ask questions about your indexed documents.")
        disabled = chunks == 0
        if disabled:
            st.caption(":red[No documents indexed. Run ingestion first.]")
        if st.button(
            "Open Chat", use_container_width=True, type="primary", disabled=disabled
        ):
            st.session_state.nav = "Chat"
            st.rerun()

elif page == "Ingestion":
    st.title("\U0001f4c4 Document Ingestion")

    if st.button("\U0001f680 Start Ingestion", type="primary"):
        with st.spinner("Ingesting documents..."):
            result = backend.ingest_documents()

        output = (result.get("stdout") or "") + "\n" + (result.get("stderr") or "")
        with st.expander("Ingestion Logs", expanded=True):
            st.code(output, language="log")

        if result.get("success"):
            st.success("Ingestion completed successfully!")
        else:
            st.error("Ingestion failed. Check logs above.")

elif page == "Chat":
    st.title("\U0001f4ac Chat with Documents")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                payload = msg["content"]
                st.markdown(payload.get("answer", ""))
                if payload.get("sources"):
                    with st.expander("\U0001f4ce Sources"):
                        for s in payload["sources"]:
                            page = (
                                f", page {s['page_number']}"
                                if s.get("page_number")
                                else ""
                            )
                            st.markdown(f"- **{s['file_name']}**{page}")
                if payload.get("context"):
                    with st.expander("\U0001f4c4 Retrieved Context"):
                        st.text(payload["context"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                result = backend.ask(prompt)
                st.markdown(result["answer"])
                if result.get("sources"):
                    with st.expander("\U0001f4ce Sources"):
                        for s in result["sources"]:
                            page = (
                                f", page {s['page_number']}"
                                if s.get("page_number")
                                else ""
                            )
                            st.markdown(f"- **{s['file_name']}**{page}")
                if result.get("context"):
                    with st.expander("\U0001f4c4 Retrieved Context"):
                        st.text(result["context"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": result}
                )
            except Exception as e:
                st.error(str(e))
                with st.expander("\U0001f50d Traceback"):
                    st.code(traceback.format_exc())
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": {
                            "answer": f"Error: {e}",
                            "sources": [],
                            "context": "",
                        },
                    }
                )
