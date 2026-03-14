"""
DocQuery — Streamlit Chat Interface
Now with: Supabase Auth, Chat Threads, Cloud File Storage
"""

import os
import sys
import re
import tempfile
from pathlib import Path

_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

from src.components.config import Config
from src.components.db import SupabaseManager
from src.components.data_ingestion import DocumentProcessor
from src.components.embeddings import EmbeddingManager
from src.components.retrieval import RetrievalManager
from src.components.genration import AnswerGenration

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="DocQuery",
    page_icon="📄",
    layout="wide",
)

# ─────────────────────────────────────────
# Init Supabase + Config (once per session)
# ─────────────────────────────────────────
if "supabase" not in st.session_state:
    st.session_state.supabase = SupabaseManager()

if "config" not in st.session_state:
    st.session_state.config = Config()

sb: SupabaseManager = st.session_state.supabase
config: Config = st.session_state.config

# ─────────────────────────────────────────
# AUTH GATE — show login if not signed in
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

    st.stop()  # Don't render the rest of the app until logged in


# ─────────────────────────────────────────
# Lazy-init RAG components (after login)
# ─────────────────────────────────────────
if "retrieval_mgr" not in st.session_state:
    # ChromaDB collection is isolated per user
    user_config = Config()
    user_config.VECTOR_DB_PATH = f"{config.VECTOR_DB_PATH}_{sb.user_id}"
    user_config.COLLECTION_NAME = f"docquery_{sb.user_id[:8]}"
    st.session_state.user_config = user_config
    st.session_state.retrieval_mgr = RetrievalManager(user_config)
    st.session_state.generator = AnswerGenration(user_config)

retrieval_mgr: RetrievalManager = st.session_state.retrieval_mgr
generator: AnswerGenration = st.session_state.generator
user_config: Config = st.session_state.user_config

# ─────────────────────────────────────────
# Sidebar — Threads + Documents
# ─────────────────────────────────────────
with st.sidebar:

    # User info + logout
    st.caption(f"👤 {sb.user_email}")
    if st.button("Sign Out", use_container_width=True):
        sb.sign_out()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.divider()

    # ── CONVERSATION THREADS ──
    st.subheader(" Conversations")


    if st.button("＋ New Chat", use_container_width=True, type="primary"):
        new_conv = sb.create_conversation("New Chat")
        st.session_state.active_conversation_id = new_conv["id"]
        st.session_state.messages = []
        st.rerun()

    conversations = sb.get_conversations()

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
                # Load messages from DB
                raw_msgs = sb.get_messages(conv["id"])
                st.session_state.messages = [
                    {"role": m["role"], "content": m["content"], "sources": m.get("sources", [])}
                    for m in raw_msgs
                ]
                st.rerun()
            if col2.button("🗑", key=f"del_conv_{conv['id']}", help="Delete thread"):
                sb.delete_conversation(conv["id"])
                if st.session_state.get("active_conversation_id") == conv["id"]:
                    st.session_state.pop("active_conversation_id", None)
                    st.session_state.messages = []
                st.rerun()

    st.divider()

    # ── DOCUMENT MANAGEMENT ──
    st.subheader(" Documents")

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
            with st.spinner(f"Uploading {uploaded_file.name}..."):
                file_bytes = uploaded_file.getvalue()
                file_ext = Path(uploaded_file.name).suffix

                # 1. Upload raw file to Supabase Storage
                storage_path = sb.upload_file(file_bytes, uploaded_file.name)

                # 2. Create document record (status=processing)
                doc_record = sb.create_document_record(
                    filename=uploaded_file.name,
                    storage_path=storage_path,
                    file_type=file_ext.strip("."),
                    file_size_bytes=uploaded_file.size,
                )
                doc_id = doc_record.get("id")

            with st.spinner("Processing & embedding..."):
                try:
                    # 3. Write to a local temp file so Unstructured can read it
                    os.makedirs(user_config.UPLOAD_DIR, exist_ok=True)
                    tmp_path = os.path.join(user_config.UPLOAD_DIR, uploaded_file.name)
                    with open(tmp_path, "wb") as f:
                        f.write(file_bytes)

                    # 4. Ingest + chunk
                    processor = DocumentProcessor(config=user_config)
                    elements = processor.process_documents(file_paths=tmp_path)

                    if elements:
                        chunks = processor.build_langchain_documents(elements=elements)

                        # 5. Embed into ChromaDB
                        embed_mgr = EmbeddingManager(config=user_config)
                        embed_mgr.create_vector_store(chunks, user_config.VECTOR_DB_PATH)

                        # 6. Update DB record to ready
                        if doc_id:
                            sb.update_document_status(doc_id, "ready", len(chunks))

                        # 7. Upload pkl caches to Supabase Storage so they survive restarts
                        for pkl_suffix in [".pkl", ".docs.pkl"]:
                            pkl_path = Path(tmp_path).with_suffix(pkl_suffix)
                            if pkl_path.exists():
                                sb.upload_pkl(pkl_path.read_bytes(), uploaded_file.name + pkl_suffix)

                        st.success(f" '{uploaded_file.name}' ready ({len(chunks)} chunks)")
                        st.session_state.retrieval_mgr = RetrievalManager(user_config)
                        st.rerun()
                    else:
                        if doc_id:
                            sb.update_document_status(doc_id, "failed")
                        st.error(" Could not extract content from this file.")
                except Exception as e:
                    if doc_id:
                        sb.update_document_status(doc_id, "failed")
                    st.error(f"Processing failed: {e}")

    # List documents from DB
    st.divider()
    docs_in_db = sb.get_user_documents()
    doc_filenames = [d["filename"] for d in docs_in_db if d["status"] == "ready"]

    if docs_in_db:
        for doc in docs_in_db:
            status_icon = {"ready": "✅", "processing": "⏳", "failed": "❌"}.get(doc["status"], "?")
            col1, col2 = st.columns([5, 1])
            col1.caption(f"{status_icon} {doc['filename']}")
            if doc["status"] == "ready":
                if col2.button("❌", key=f"deldoc_{doc['id']}", help="Delete"):
                    with st.spinner("Deleting..."):
                        # Remove from ChromaDB
                        retrieval_mgr.delete_document_by_filename(doc["filename"])
                        # Remove from Supabase Storage
                        sb.delete_file(doc["storage_path"])
                        # Remove DB record
                        sb.delete_document_record(doc["filename"])
                        st.toast(f"Deleted {doc['filename']}")
                        st.rerun()
    else:
        st.caption("No documents yet.")

    st.divider()

    # Document filter for search
    selected_filter = st.selectbox(
        "Search in",
        ["All Documents"] + doc_filenames,
    )
    filename_filter = selected_filter if selected_filter != "All Documents" else None


# ─────────────────────────────────────────
# Main Chat Area
# ─────────────────────────────────────────

# If no active conversation, prompt user to start one
if "active_conversation_id" not in st.session_state:
    st.title(" DocQuery")
    st.info("Start a **New Chat** from the sidebar, or select an existing conversation.")
    st.stop()

active_conv_id = st.session_state.active_conversation_id

# Find title of active conversation
active_conv = next((c for c in conversations if c["id"] == active_conv_id), None)
conv_title = active_conv["title"] if active_conv else "Chat"

st.title(f" {conv_title}")
st.caption(f"Searching: **{selected_filter}**")

# Render message history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"View Sources ({len(message['sources'])})"):
                for idx, src in enumerate(message["sources"], 1):
                    if isinstance(src, dict):
                        st.markdown(f"**Source {idx}:** {src.get('filename', 'Unknown')} (Page {src.get('page', 'N/A')})")
                    else:
                        meta = getattr(src, "metadata", {})
                        st.markdown(f"**Source {idx}:** {meta.get('filename', 'Unknown')} (Page {meta.get('page_number', 'N/A')})")
                        st.caption(f"_{src.page_content[:300]}_")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Auto-title conversation on first message
    if len(st.session_state.messages) == 0:
        sb.auto_title_conversation(active_conv_id, prompt)

    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})

    # Persist user message to DB
    sb.save_message(active_conv_id, "user", prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving..."):
            page_filter = None
            page_match = re.search(r'page\s+(\d+)', prompt.lower())
            if page_match:
                page_filter = int(page_match.group(1))
            docs = retrieval_mgr.retrieve(prompt, filename_filter=filename_filter, page_filter=page_filter)

        if not docs:
            answer = "I couldn't find relevant information in your documents for this question."
            st.warning(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": []})
            sb.save_message(active_conv_id, "assistant", answer)
        else:
            try:
                stream, sources = generator.generate_stream(prompt, docs,chat_history=st.session_state.messages[:-1])
                answer = st.write_stream(stream)

                # Show sources
                with st.expander(f"View Sources ({len(docs)})"):
                    for idx, src in enumerate(docs, 1):
                        meta = src.metadata if hasattr(src, "metadata") else {}
                        st.markdown(f"**Source {idx}:** {meta.get('filename', 'Unknown')} (Page {meta.get('page_number', 'N/A')})")
                        st.caption(f"_{src.page_content[:300]}_")

                # Serialize sources for DB storage (only serializable fields)
                serializable_sources = [
                    {
                        "filename": s.get("filename"),
                        "page": s.get("page"),
                        "chunk_type": s.get("chunk_type"),
                    }
                    for s in sources
                ]

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": serializable_sources,
                })

                # Persist assistant message to DB
                sb.save_message(active_conv_id, "assistant", answer, serializable_sources)

            except Exception as e:
                err = f"Error generating response: `{e}`"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err, "sources": []})