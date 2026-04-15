import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

import chatbot
import feedback_store

load_dotenv()

LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "qa-chatbot")
HISTORY_FILE = Path(__file__).parent / "chat_history.json"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="QA Assistant", layout="wide")

# ── Custom CSS + global copy JS (injected once) ───────────────────────────────
st.markdown("""
<style>
.qa-card { background:#1e1e2e; border:1px solid #313244; border-radius:12px;
           padding:16px 18px; height:100%; }
.qa-card:hover { border-color:#89b4fa; background:#252535; }
.stChatMessage { margin-bottom:4px; }

.copy-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: transparent;
    border: 1px solid #444;
    border-radius: 6px;
    color: #888;
    cursor: pointer;
    font-size: 12px;
    padding: 3px 10px;
    margin-top: 6px;
    transition: color 0.2s, border-color 0.2s;
}
.copy-btn:hover { color: #cdd6f4; border-color: #89b4fa; }
.copy-btn svg { width:13px; height:13px; fill:currentColor; }
</style>

<script>
function qaCopy(encodedText, btn) {
    const text = decodeURIComponent(encodedText);
    navigator.clipboard.writeText(text).then(function() {
        btn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M9 16.2L4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4L9 16.2z"/></svg> Copied!';
        btn.style.color = '#a6e3a1';
        btn.style.borderColor = '#a6e3a1';
        setTimeout(function() {
            btn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M16 1H4C3 1 2 2 2 3v14h2V3h12V1zm3 4H8C7 5 6 6 6 7v14c0 1 1 2 2 2h11c1 0 2-1 2-2V7c0-1-1-2-2-2zm0 16H8V7h11v14z"/></svg> Copy';
            btn.style.color = '';
            btn.style.borderColor = '';
        }, 2000);
    });
}
</script>
""", unsafe_allow_html=True)


# ── Copy helper ───────────────────────────────────────────────────────────────

def copy_button(text: str, key: str) -> None:
    """Self-contained iframe copy button using execCommand — works inside Streamlit iframes."""
    safe = json.dumps(text)   # properly escaped: handles quotes, newlines, unicode
    components.html(
        f"""<!DOCTYPE html>
<html>
<head>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: transparent; font-family: -apple-system, sans-serif; padding: 2px 0; }}
  button {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: transparent;
    border: 1px solid #555;
    border-radius: 6px;
    color: #999;
    cursor: pointer;
    font-size: 12px;
    padding: 3px 10px;
    transition: color 0.15s, border-color 0.15s;
  }}
  button:hover {{ color: #cdd6f4; border-color: #89b4fa; }}
  button.copied {{ color: #a6e3a1; border-color: #a6e3a1; }}
  svg {{ width: 13px; height: 13px; fill: currentColor; flex-shrink: 0; }}
  #ta {{ position: fixed; top: -9999px; left: -9999px; opacity: 0; }}
</style>
</head>
<body>
<textarea id="ta" readonly></textarea>
<button id="btn" onclick="doCopy()">
  <svg viewBox="0 0 24 24"><path d="M16 1H4C3 1 2 2 2 3v14h2V3h12V1zm3 4H8C7 5 6 6 6 7v14c0 1 1 2 2 2h11c1 0 2-1 2-2V7c0-1-1-2-2-2zm0 16H8V7h11v14z"/></svg>
  Copy
</button>
<script>
var TEXT = {safe};
function doCopy() {{
  var ta = document.getElementById('ta');
  ta.value = TEXT;
  ta.select();
  ta.setSelectionRange(0, 999999);
  var ok = document.execCommand('copy');
  var b = document.getElementById('btn');
  if (ok) {{
    b.classList.add('copied');
    b.innerHTML = '<svg viewBox="0 0 24 24"><path d="M9 16.2L4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4L9 16.2z"/></svg> Copied!';
    setTimeout(function() {{
      b.classList.remove('copied');
      b.innerHTML = '<svg viewBox="0 0 24 24"><path d="M16 1H4C3 1 2 2 2 3v14h2V3h12V1zm3 4H8C7 5 6 6 6 7v14c0 1 1 2 2 2h11c1 0 2-1 2-2V7c0-1-1-2-2-2zm0 16H8V7h11v14z"/></svg> Copy';
    }}, 2000);
  }}
}}
</script>
</body>
</html>""",
        height=34,
        scrolling=False,
    )


# ── Per-message feedback widget ───────────────────────────────────────────────

def render_feedback_widget(msg_idx: int, assistant_text: str, user_text: str) -> None:
    """Render 👍/👎 + optional comment for a single assistant message.

    Tracks per-message state in st.session_state under key f"fb_{msg_idx}".
    Stages: 'initial' → 'awaiting_comment' → 'done'
    """
    state_key = f"fb_{msg_idx}"
    if state_key not in st.session_state:
        st.session_state[state_key] = {"stage": "initial"}

    stage = st.session_state[state_key].get("stage", "initial")

    if stage == "done":
        status = st.session_state[state_key].get("status", "")
        if status in ("saved", "skipped_duplicate"):
            st.caption("✅ Feedback saved")
        elif status == "skipped_validation":
            st.caption("✅ Noted")
        else:
            st.caption("❌ Failed to save (retry)")
        return

    if stage == "initial":
        col1, col2, _ = st.columns([1, 1, 10])
        with col1:
            if st.button("👍", key=f"up_{msg_idx}", help="Good response"):
                try:
                    store = st.session_state.feedback
                    result = feedback_store.add_positive(
                        user_text,
                        assistant_text,
                        store,
                        session_id=st.session_state.session_id,
                    )
                    st.session_state.feedback = store
                    st.session_state[state_key] = {"stage": "done", "status": result}
                except Exception:
                    st.session_state[state_key] = {"stage": "done", "status": "error"}
                st.rerun()
        with col2:
            if st.button("👎", key=f"down_{msg_idx}", help="Poor response"):
                st.session_state[state_key]["stage"] = "awaiting_comment"
                st.rerun()

    elif stage == "awaiting_comment":
        comment = st.text_input(
            "What was wrong? (optional)",
            key=f"comment_{msg_idx}",
            placeholder="e.g. missed boundary values on the password field",
        )
        col1, col2, _ = st.columns([1, 1, 8])
        with col1:
            if st.button("Submit", key=f"submit_{msg_idx}", type="primary"):
                try:
                    store = st.session_state.feedback
                    result = feedback_store.add_negative(
                        comment,
                        user_text,
                        assistant_text,
                        store,
                        session_id=st.session_state.session_id,
                    )
                    st.session_state.feedback = store
                    st.session_state[state_key] = {"stage": "done", "status": result}
                except Exception:
                    st.session_state[state_key] = {"stage": "done", "status": "error"}
                st.rerun()
        with col2:
            if st.button("Cancel", key=f"cancel_{msg_idx}"):
                st.session_state[state_key]["stage"] = "initial"
                st.rerun()


# ── History helpers ───────────────────────────────────────────────────────────

def load_all_sessions() -> dict:
    """Load all sessions from disk. Returns {} if file missing or corrupt."""
    if not HISTORY_FILE.exists():
        return {}
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_all_sessions(sessions: dict) -> None:
    """Persist all sessions to disk."""
    HISTORY_FILE.write_text(
        json.dumps(sessions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def session_title(session: dict) -> str:
    """First user message (truncated) as the display title."""
    for msg in session.get("messages", []):
        if msg["role"] == "user":
            text = msg["content"].strip().replace("\n", " ")
            return text[:55] + "…" if len(text) > 55 else text
    return "Empty session"


def save_current_session() -> None:
    """Write the active session back into the history file."""
    if not st.session_state.messages:
        return
    sessions = load_all_sessions()
    sid = st.session_state.session_id
    sessions[sid] = {
        "id": sid,
        "created_at": st.session_state.session_created_at,
        "messages": st.session_state.messages,
        "total_messages": st.session_state.total_messages,
    }
    save_all_sessions(sessions)


def new_session() -> None:
    """Save current session and initialise a fresh one."""
    save_current_session()
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.session_created_at = datetime.now(timezone.utc).isoformat()
    st.session_state.messages = []
    st.session_state.total_messages = 0
    st.session_state.prefill = ""


def load_session(sid: str) -> None:
    """Save current session then switch to a historical one."""
    save_current_session()
    sessions = load_all_sessions()
    s = sessions.get(sid, {})
    st.session_state.session_id = sid
    st.session_state.session_created_at = s.get("created_at", "")
    st.session_state.messages = s.get("messages", [])
    st.session_state.total_messages = s.get("total_messages", 0)
    st.session_state.prefill = ""


# ── Session state init ────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "session_created_at" not in st.session_state:
    st.session_state.session_created_at = datetime.now(timezone.utc).isoformat()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_messages" not in st.session_state:
    st.session_state.total_messages = 0
if "prefill" not in st.session_state:
    st.session_state.prefill = ""
if "feedback" not in st.session_state:
    st.session_state.feedback = feedback_store.load_store()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("QA Assistant")

    if st.button("＋  New Chat", use_container_width=True, type="primary"):
        new_session()
        st.rerun()

    st.divider()

    # ── Chat history list ─────────────────────────────────────────────────────
    all_sessions = load_all_sessions()

    # Include the active (possibly unsaved) session in the list
    active_id = st.session_state.session_id
    if st.session_state.messages and active_id not in all_sessions:
        all_sessions[active_id] = {
            "id": active_id,
            "created_at": st.session_state.session_created_at,
            "messages": st.session_state.messages,
            "total_messages": st.session_state.total_messages,
        }

    # Sort newest first
    sorted_sessions = sorted(
        all_sessions.values(),
        key=lambda s: s.get("created_at", ""),
        reverse=True,
    )

    if sorted_sessions:
        st.markdown("**Chat History**")
        for s in sorted_sessions:
            sid = s["id"]
            title = session_title(s)
            is_active = sid == active_id

            col_btn, col_del = st.columns([5, 1])
            with col_btn:
                label = f"**→ {title}**" if is_active else title
                if st.button(label, key=f"load_{sid}", use_container_width=True):
                    if not is_active:
                        load_session(sid)
                        st.rerun()
            with col_del:
                if st.button("🗑", key=f"del_{sid}", help="Delete this chat"):
                    sessions = load_all_sessions()
                    sessions.pop(sid, None)
                    save_all_sessions(sessions)
                    if is_active:
                        new_session()
                    st.rerun()
    else:
        st.caption("No saved chats yet.")

    st.divider()
    st.markdown("**Session ID**")
    st.code(active_id, language=None)
    st.markdown(f"**Messages sent:** {st.session_state.total_messages}")
    langsmith_url = f"https://smith.langchain.com/projects/{LANGSMITH_PROJECT}"
    st.markdown(f"[View traces in LangSmith]({langsmith_url})")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("QA Assistant")
st.caption("Senior QA engineer · banking & e-commerce · 10+ years experience")

# ── Quick-action cards (shown only when chat is empty) ────────────────────────
QUICK_ACTIONS = [
    {
        "icon": "🧪",
        "title": "Write Test Cases",
        "desc": "Generate structured test cases from a user story or requirement",
        "prompt": "Write test cases for the following user story:\n\n",
    },
    {
        "icon": "📖",
        "title": "Write User Story",
        "desc": "Turn a feature idea into a well-formed user story with ACs",
        "prompt": "Write a user story with acceptance criteria for:\n\n",
    },
    {
        "icon": "🔍",
        "title": "Identify Edge Cases",
        "desc": "Find boundary conditions and edge cases for a feature",
        "prompt": "Identify edge cases and boundary conditions for:\n\n",
    },
    {
        "icon": "📋",
        "title": "Review Test Plan",
        "desc": "Get expert feedback on an existing test plan",
        "prompt": "Review the following test plan and give structured feedback:\n\n",
    },
    {
        "icon": "🐛",
        "title": "Write Bug Report",
        "desc": "Format a bug report with steps to reproduce and severity",
        "prompt": "Write a detailed bug report for the following issue:\n\n",
    },
    {
        "icon": "🗺️",
        "title": "Testing Strategy",
        "desc": "Suggest a testing strategy (functional, regression, exploratory)",
        "prompt": "Suggest a testing strategy for:\n\n",
    },
    {
        "icon": "📝",
        "title": "Test Case Titles Only",
        "desc": "Get a quick list of test case titles without full detail",
        "prompt": (
            "List ONLY the test case titles for the following requirement. "
            "No steps, no expected results — titles only. "
            "Each title must follow the pattern: "
            "\"Verify that [subject] [behaviour] when [condition]\"\n\n"
        ),
    },
]

if not st.session_state.messages:
    st.markdown("#### What would you like to do?")
    cols = st.columns(3)
    for idx, action in enumerate(QUICK_ACTIONS):
        with cols[idx % 3]:
            if st.button(
                f"{action['icon']} **{action['title']}**\n\n{action['desc']}",
                key=f"card_{idx}",
                use_container_width=True,
                help=action["desc"],
            ):
                st.session_state.prefill = action["prompt"]
                st.rerun()
    st.divider()

# ── File upload ───────────────────────────────────────────────────────────────
with st.expander("📎 Attach a file (optional — plain text or .md)", expanded=False):
    uploaded_file = st.file_uploader(
        label="Upload file",
        type=["txt", "md", "csv", "json", "yaml", "yml"],
        label_visibility="collapsed",
    )
    file_content = ""
    if uploaded_file is not None:
        try:
            raw = uploaded_file.read()
            file_content = raw.decode("utf-8")
            st.success(f"Loaded **{uploaded_file.name}** ({len(file_content.split())} words)")
            with st.expander("Preview", expanded=False):
                st.text(file_content[:2000] + ("…" if len(file_content) > 2000 else ""))
        except UnicodeDecodeError:
            st.error(
                f"Could not read **{uploaded_file.name}** as text. "
                "Please upload a plain-text file (UTF-8 encoded)."
            )
        except Exception as exc:
            st.error(f"Unexpected error reading file: {exc}")

# ── Chat history render ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        copy_button(msg["content"], key=f"copy_history_{id(msg)}")
        if msg["role"] == "assistant" and "input_tokens" in msg:
            st.caption(
                f"Tokens — input: {msg['input_tokens']} | output: {msg['output_tokens']}"
            )

# ── Chat input ────────────────────────────────────────────────────────────────
placeholder_text = (
    "Paste your story / requirement here…"
    if st.session_state.prefill
    else "Ask a testing question or paste a user story…"
)

user_input = st.chat_input(placeholder_text)

# Quick-action pre-fill editor
if st.session_state.prefill and not user_input:
    st.info("Complete the prompt below and press **Send**.")
    draft = st.text_area(
        "Your prompt",
        value=st.session_state.prefill,
        height=140,
        key="prefill_area",
    )
    send_col, cancel_col = st.columns([1, 5])
    with send_col:
        send_clicked = st.button("Send", type="primary")
    with cancel_col:
        if st.button("Cancel"):
            st.session_state.prefill = ""
            st.rerun()
    if send_clicked:
        user_input = draft
        st.session_state.prefill = ""

# ── Process message ───────────────────────────────────────────────────────────
if user_input:
    if file_content:
        full_input = (
            f"{user_input}\n\n---\n"
            f"**Attached file: {uploaded_file.name}**\n"
            f"```\n{file_content}\n```"
        )
    else:
        full_input = user_input

    st.session_state.messages.append({"role": "user", "content": full_input})
    st.session_state.total_messages += 1

    with st.chat_message("user"):
        st.markdown(full_input)
        copy_button(full_input, key="copy_user_input")

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
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
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"_(Error: {result['error']})_",
            })
        else:
            st.markdown(result["text"])
            copy_button(result["text"], key="copy_assistant_response")
            st.caption(
                f"Tokens — input: {result['input_tokens']} | output: {result['output_tokens']}"
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["text"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
            })

    # Auto-save after every exchange
    save_current_session()
    st.rerun()
