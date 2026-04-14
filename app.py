import os
import uuid
from datetime import datetime, timezone

import streamlit as st
from dotenv import load_dotenv

import chatbot

load_dotenv()

LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "qa-chatbot")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="QA Assistant", layout="wide")

# ── Session state init ────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_messages" not in st.session_state:
    st.session_state.total_messages = 0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("QA Assistant")
    st.divider()

    st.markdown("**Session ID**")
    st.code(st.session_state.session_id, language=None)

    st.markdown(f"**Messages sent:** {st.session_state.total_messages}")

    langsmith_url = f"https://smith.langchain.com/projects/{LANGSMITH_PROJECT}"
    st.markdown(f"[View traces in LangSmith]({langsmith_url})")

    st.divider()
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_messages = 0
        st.rerun()

# ── Main chat area ─────────────────────────────────────────────────────────────
st.title("QA Assistant")
st.caption("Senior QA engineer with 10+ years of experience. Ask me about test cases, edge cases, and testing strategies.")

# Render existing conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "input_tokens" in msg:
            st.caption(
                f"Tokens — input: {msg['input_tokens']} | output: {msg['output_tokens']}"
            )

# ── Input ─────────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask a testing question..."):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.total_messages += 1

    with st.chat_message("user"):
        st.markdown(user_input)

    # Call Claude via chatbot.py, passing LangSmith metadata
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chatbot.get_response(
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                session_id=st.session_state.session_id,
                langsmith_extra={
                    "metadata": {
                        "session_id": st.session_state.session_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "project": LANGSMITH_PROJECT,
                    }
                },
            )

        if "error" in result:
            st.error(f"Error communicating with Claude: {result['error']}")
            # Store a placeholder so history stays in sync with total_messages
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"_(Error: {result['error']})_",
            })
        else:
            st.markdown(result["text"])
            st.caption(
                f"Tokens — input: {result['input_tokens']} | output: {result['output_tokens']}"
            )
            # Store with token counts for re-rendering history
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["text"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
            })
