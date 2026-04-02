"""
Streamlit chat UI.
Calls the FastAPI backend instead of generate.py directly.
"""

import os
import json
import requests
import streamlit as st
from pathlib import Path

API_URL    = os.environ.get("API_URL", "http://localhost:8000")
DISCLAIMER = "⚠️ This is not medical advice. Consult a qualified healthcare professional for diagnosis and treatment."


def strip_disclaimer(text: str) -> str:
    return text.replace(DISCLAIMER, "").strip()

st.set_page_config(
    page_title="Medical Q&A Assistant",
    page_icon="🏥",
    layout="wide",
)

st.title("Medical Q&A Assistant")
st.info(DISCLAIMER)

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Retrieved records (top-k)", min_value=1, max_value=10, value=5)
    show_sources = st.toggle("Show sources", value=True)

    st.divider()
    st.header("Session Stats")
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    if "total_latency" not in st.session_state:
        st.session_state.total_latency = 0

    st.metric("Queries this session", st.session_state.total_queries)
    if st.session_state.total_queries > 0:
        avg_latency = st.session_state.total_latency // st.session_state.total_queries
        st.metric("Avg latency", f"{avg_latency}ms")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.total_latency = 0
        st.rerun()


def stream_from_api(query: str, top_k: int):
    """
    Calls POST /ask/stream and yields text chunks, then a final metadata dict.
    Mirrors the interface of generate.stream_answer() so app logic is unchanged.
    """
    with requests.post(
        f"{API_URL}/ask/stream",
        json={"query": query, "top_k": top_k},
        stream=True,
        timeout=60,
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                return
            data = json.loads(payload)
            if data["type"] == "text":
                yield data["content"]
            elif data["type"] == "meta":
                yield {k: v for k, v in data.items() if k != "type"}


# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])}) — {msg['latency_ms']}ms"):
                for s in msg["sources"]:
                    st.markdown(
                        f"**[{s['similarity']}]** `{s['task']}`  \n{s['question']}"
                    )

# --- Input ---
if query := st.chat_input("Ask a clinical question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving records..."):
            gen = stream_from_api(query, top_k=top_k)
            first = next(gen)

        result = None
        if isinstance(first, dict):
            result = first
            result["answer"] = strip_disclaimer(result["answer"])
            st.markdown(result["answer"])
        else:
            def token_stream():
                yield first
                for chunk in gen:
                    if isinstance(chunk, dict):
                        token_stream.meta = chunk
                    else:
                        yield chunk
            token_stream.meta = None

            full_text = st.write_stream(token_stream())
            result = token_stream.meta
            result["answer"] = strip_disclaimer(full_text)

        if show_sources and result["sources"]:
            with st.expander(f"Sources ({len(result['sources'])}) — {result['latency_ms']}ms"):
                for s in result["sources"]:
                    st.markdown(
                        f"**[{s['similarity']}]** `{s['task']}`  \n{s['question']}"
                    )

    st.session_state.messages.append({
        "role":       "assistant",
        "content":    result["answer"],
        "sources":    result["sources"],
        "latency_ms": result["latency_ms"],
    })

    st.session_state.total_queries += 1
    st.session_state.total_latency += result["latency_ms"]
    st.rerun()
