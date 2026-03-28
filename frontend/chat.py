"""
DocQuery — Streamlit Chat Interface (API Client Mode)

Streamlit is a THIN CLIENT here. It:
  1. Handles auth via Supabase directly (login/logout only)
  2. Proxies ALL RAG operations to the FastAPI backend via httpx

Set API_BASE_URL in your .env to point at the running FastAPI server.
Default: http://localhost:8000
"""

import os
import sys
import re
import json
from pathlib import Path

_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import httpx
import streamlit as st

from src.components.config import Config
from src.components.db import SupabaseManager

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="DocQuery",
    page_icon="📄",
    layout="wide",
)

# ─────────────────────────────────────────
# Config & Supabase (auth only)
# ─────────────────────────────────────────
if "config" not in st.session_state:
    st.session_state.config = Config()

if "supabase" not in st.session_state:
    st.session_state.supabase = SupabaseManager()

sb: SupabaseManager = st.session_state.supabase
config: Config = st.session_state.config

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
API_V1 = f"{API_BASE_URL}/api/v1"


def _auth_headers() -> dict:
    """Return Authorization header using the current Supabase session token."""
    try:
        session = sb.client.auth.get_session()
        if session and session.access_token:
            return {"Authorization": f"Bearer {session.access_token}"}
    except Exception:
        pass
    return {}


def api_get(path: str, **kwargs):
    return httpx.get(f"{API_V1}{path}", headers=_auth_headers(), timeout=30, **kwargs)


def api_post(path: str, **kwargs):
    return httpx.post(f"{API_V1}{path}", headers=_auth_headers(), timeout=120, **kwargs)


def api_delete(path: str, **kwargs):
    return httpx.delete(f"{API_V1}{path}", headers=_auth_headers(), timeout=30, **kwargs)


# ─────────────────────────────────────────
# AUTH GATE
# ─────────────────────────────────────────
user = sb.get_user()

if not user:
    st.title("📄 DocQuery")
    st.caption("Sign in to start querying your documents.")

    tab_login, tab_signup = st.tabs(["Sign In", "Sign Up"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In", type="primary")
            if submitted:
                try:
                    sb.sign_in(email, password)
                    st.rerun()
                except Exception as e:
                    st.error(f"Login failed: {e}")

    with tab_signup:
        with st.form("signup_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Create Account", type="primary")
            if submitted:
                try:
                    sb.sign_up(email, password)
                    st.success("Account created! Please sign in.")
                except Exception as e:
                    st.error(f"Sign up failed: {e}")

    st.stop()


# ─────────────────────────────────────────
# Sidebar — Threads + Documents
# ─────────────────────────────────────────
with st.sidebar:
    st.caption(f"👤 {sb.user_email}")
    if st.button("Sign Out", use_container_width=True):
        sb.sign_out()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.divider()

    # ── CONVERSATIONS ──
    st.subheader("💬 Conversations")

    if st.button("＋ New Chat", use_container_width=True, type="primary"):
        resp = api_post("/conversations", json={"title": "New Chat"})
        if resp.status_code == 200:
            new_conv = resp.json()
            st.session_state.active_conversation_id = new_conv["id"]
            st.session_state.messages = []
            st.rerun()
        else:
            st.error("Could not create conversation.")

    conv_resp = api_get("/conversations")
    conversations = conv_resp.json().get("conversations", []) if conv_resp.status_code == 200 else []

    if not conversations:
        st.caption("No conversations yet. Start a new chat!")
    else:
        active_id = st.session_state.get("active_conversation_id")
        for conv in conversations:
            col1, col2 = st.columns([5, 1])
            is_active = conv["id"] == active_id
            label = f"**{conv['title']}**" if is_active else conv["title"]
            if col1.button(label, key=f"conv_{conv['id']}", use_container_width=True):
                st.session_state.active_conversation_id = conv["id"]
                msg_resp = api_get(f"/conversations/{conv['id']}/messages")
                if msg_resp.status_code == 200:
                    raw_msgs = msg_resp.json().get("messages", [])
                    st.session_state.messages = [
                        {"role": m["role"], "content": m["content"], "sources": m.get("sources", [])}
                        for m in raw_msgs
                    ]
                else:
                    st.session_state.messages = []
                st.rerun()
            if col2.button("🗑", key=f"del_conv_{conv['id']}", help="Delete thread"):
                api_delete(f"/conversations/{conv['id']}")
                if st.session_state.get("active_conversation_id") == conv["id"]:
                    st.session_state.pop("active_conversation_id", None)
                    st.session_state.messages = []
                st.rerun()

    st.divider()

    # ── DOCUMENTS ──
    st.subheader("📁 Documents")

    MAX_FILE_SIZE_MB = 10
    supported_exts = list(config.SUPPORTED_FILE_TYPES)
    uploaded_file = st.file_uploader(
        f"Upload (max {MAX_FILE_SIZE_MB}MB)",
        type=supported_exts,
    )

    if uploaded_file:
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"File too large (max {MAX_FILE_SIZE_MB}MB).")
        elif st.button("Process Document", type="primary"):
            with st.spinner(f"Uploading & processing {uploaded_file.name}…"):
                try:
                    file_bytes = uploaded_file.getvalue()
                    resp = api_post(
                        "/documents/upload",
                        files={"file": (uploaded_file.name, file_bytes, "application/octet-stream")},
                    )
                    if resp.status_code in (200, 202):
                        doc = resp.json()
                        if doc.get("status") == "processing":
                            st.info(
                                f"⏳ '{doc['filename']}' is being processed in the background. "
                                "The document list below will update when ready."
                            )
                        else:
                            st.success(
                                f"✅ '{doc['filename']}' ready ({doc.get('chunk_count', 0)} chunks)"
                            )
                        st.rerun()
                    else:
                        detail = resp.json().get("detail", resp.text)
                        st.error(f"Upload failed: {detail}")
                except Exception as e:
                    st.error(f"Upload error: {e}")

    st.divider()

    docs_resp = api_get("/documents")
    docs_in_db = docs_resp.json().get("documents", []) if docs_resp.status_code == 200 else []
    doc_filenames = [d["filename"] for d in docs_in_db if d["status"] == "ready"]

    if docs_in_db:
        for doc in docs_in_db:
            status_icon = {"ready": "✅", "processing": "⏳", "failed": "❌"}.get(doc["status"], "?")
            col1, col2 = st.columns([5, 1])
            col1.caption(f"{status_icon} {doc['filename']}")
            if doc["status"] == "ready":
                if col2.button("❌", key=f"deldoc_{doc['id']}", help="Delete"):
                    with st.spinner("Deleting…"):
                        api_delete(f"/documents/{doc['filename']}")
                        st.toast(f"Deleted {doc['filename']}")
                        st.rerun()
    else:
        st.caption("No documents yet.")

    st.divider()

    selected_filter = st.selectbox(
        "Search in",
        ["All Documents"] + doc_filenames,
    )
    filename_filter = selected_filter if selected_filter != "All Documents" else None


# ─────────────────────────────────────────
# Main Chat Area
# ─────────────────────────────────────────
if "active_conversation_id" not in st.session_state:
    st.title("📄 DocQuery")
    st.info("Start a **New Chat** from the sidebar, or select an existing conversation.")
    st.stop()

active_conv_id = st.session_state.active_conversation_id
active_conv = next((c for c in conversations if c["id"] == active_conv_id), None)
conv_title = active_conv["title"] if active_conv else "Chat"

st.title(f"💬 {conv_title}")
st.caption(f"Searching: **{selected_filter}**")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"View Sources ({len(message['sources'])})"):
                for idx, src in enumerate(message["sources"], 1):
                    st.markdown(
                        f"**Source {idx}:** {src.get('filename', 'Unknown')} "
                        f"(Page {src.get('page', 'N/A')})"
                    )
                    content_snippet = src.get("content")
                    if content_snippet:
                        st.markdown(f"> _{content_snippet}_")


# ─────────────────────────────────────────
# Chat input
# ─────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documents…"):
    # Show user message immediately
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})

    # Persist user message directly to Supabase (avoids a second LLM call later)
    sb.save_message(active_conv_id, "user", prompt)

    # Auto-title conversation on first message
    if len(st.session_state.messages) == 1:
        sb.auto_title_conversation(active_conv_id, prompt)

    # Detect page number filter
    page_filter = None
    page_match = re.search(r'page\s+(\d+)', prompt.lower())
    if page_match:
        page_filter = int(page_match.group(1))

    payload = {
        "question": prompt,
        "filename_filter": filename_filter,
        "page_filter": page_filter,
        "conversation_id": active_conv_id,
    }

    with st.chat_message("assistant"):
        try:
            headers = _auth_headers()
            headers["Accept"] = "text/event-stream"

            answer_placeholder = st.empty()
            full_answer = ""
            sources = []

            # Stream response from FastAPI SSE endpoint
            with httpx.Client(timeout=120) as client:
                with client.stream(
                    "POST",
                    f"{API_V1}/query/stream",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status_code != 200:
                        err_body = response.read().decode()
                        st.error(f"API error {response.status_code}: {err_body}")
                    else:
                        for line in response.iter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            data_str = line[6:]  # strip "data: " prefix
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                if data.get("type") == "token":
                                    full_answer += data.get("content", "")
                                    answer_placeholder.markdown(full_answer + "▌")
                                elif data.get("type") == "sources":
                                    sources = data.get("sources", [])
                                elif data.get("type") == "error":
                                    st.error(data.get("message", "Unknown error"))
                            except json.JSONDecodeError:
                                pass

                        answer_placeholder.markdown(full_answer)

            if full_answer:
                # Persist assistant answer directly to Supabase — no second LLM call
                sb.save_message(active_conv_id, "assistant", full_answer, sources)

                if sources:
                    with st.expander(f"View Sources ({len(sources)})"):
                        for idx, src in enumerate(sources, 1):
                            st.markdown(
                                f"**Source {idx}:** {src.get('filename', 'Unknown')} "
                                f"(Page {src.get('page', 'N/A')})"
                            )
                            content_snippet = src.get("content")
                            if content_snippet:
                                st.markdown(f"> _{content_snippet}_")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_answer,
                    "sources": sources,
                })

        except Exception as e:
            err = f"Error: `{e}`"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err, "sources": []})