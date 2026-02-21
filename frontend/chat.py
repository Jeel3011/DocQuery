import os
import sys
from pathlib import Path
import shutil

# Add project root to sys.path so we can import from src
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from src.components.config import Config
from src.components.data_ingestion import DocumentProcessor
from src.components.embeddings import EmbeddingManager
from src.components.retrieval import RetrievalManager
from src.components.genration import AnswerGenration

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="DocQuery Chat",
    page_icon="📄",
    layout="wide"  # Changed to wide to fit better
)

# ==========================================
# Session & Multi-User Management
# ==========================================
if "user_id" not in st.session_state:
    st.session_state.user_id = "default"

with st.sidebar:
    st.header("👤 Workspace Session")
    new_user_id = st.text_input("Session ID (User)", value=st.session_state.user_id, help="Change this to create isolated workspaces.")
    if new_user_id != st.session_state.user_id:
        st.session_state.user_id = new_user_id
        # Reset state for new workspace
        for key in ["config", "retrieval_mgr", "generator", "messages"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    st.divider()

# ==========================================
# Initialize Components
# ==========================================
if "config" not in st.session_state:
    cfg = Config()
    # Isolate uploading directories and db path to multi-user sessions
    cfg.UPLOAD_DIR = os.path.join(cfg.UPLOAD_DIR, st.session_state.user_id)
    cfg.VECTOR_DB_PATH = f"{cfg.VECTOR_DB_PATH}_{st.session_state.user_id}"
    st.session_state.config = cfg

if "retrieval_mgr" not in st.session_state:
    st.session_state.retrieval_mgr = RetrievalManager(st.session_state.config)

if "generator" not in st.session_state:
    st.session_state.generator = AnswerGenration(st.session_state.config)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Ensure upload directory exists
os.makedirs(st.session_state.config.UPLOAD_DIR, exist_ok=True)

# ==========================================
# Sidebar: Document Management
# ==========================================
with st.sidebar:
    st.header("📄 Document Management")
    
    # List supported extensions dynamically
    supported_exts = [ext.replace(".", "") for ext in st.session_state.config.SUPPORTED_FILE_TYPES]
    
    MAX_FILE_SIZE_MB = 10
    uploaded_file = st.file_uploader(
        f"Upload a document (Max {MAX_FILE_SIZE_MB}MB)", 
        type=supported_exts
    )
    
    if uploaded_file:
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"File size exceeds the {MAX_FILE_SIZE_MB}MB limit.")
        elif st.button("Process Document", type="primary"):
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                # 1. Save uploaded file
                file_path = os.path.join(st.session_state.config.UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                # 2. Process File
                processor = DocumentProcessor(config=st.session_state.config)
                elements = processor.process_documents(file_paths=file_path)
                
                if elements:
                    # 3. Chunk
                    st.info("Files partitioned. Chunking and building embeddings...")
                    chunks = processor.build_langchain_documents(elements=elements)
                    
                    # 4. Embed
                    embed_mgr = EmbeddingManager(config=st.session_state.config)
                    embed_mgr.create_vector_store(chunks, st.session_state.config.VECTOR_DB_PATH)
                    
                    st.success(f"✅ Successfully added '{uploaded_file.name}'!")
                    # Refresh retrieval manager since DB was updated
                    st.session_state.retrieval_mgr = RetrievalManager(st.session_state.config)
                    st.rerun()
                else:
                    st.error("❌ Failed to process document. See logs for details.")

    st.divider()
    st.subheader("Uploaded Documents")
    
    # Calculate visible docs
    supported_exts_with_dot = [f".{ext}" for ext in supported_exts]
    doc_files = [
        f for f in os.listdir(st.session_state.config.UPLOAD_DIR) 
        if any(f.endswith(ext) for ext in supported_exts_with_dot) and not f.startswith('.')
    ]
    doc_files.sort()
    
    if doc_files:
        for f in doc_files:
            col1, col2 = st.columns([5, 1])
            col1.caption(f)
            if col2.button("❌", key=f"del_{f}", help="Delete document"):
                with st.spinner("Deleting..."):
                    # 1. Remove from vector db
                    st.session_state.retrieval_mgr.delete_document_by_filename(f)
                    
                    # 2. Remove files 
                    base_path = os.path.join(st.session_state.config.UPLOAD_DIR, f)
                    if os.path.exists(base_path): 
                        os.remove(base_path)
                    
                    # 3. Remove Cache
                    cache1 = Path(base_path).with_suffix('.pkl')
                    cache2 = Path(base_path).with_suffix('.docs.pkl')
                    if cache1.exists(): os.remove(cache1)
                    if cache2.exists(): os.remove(cache2)
                    
                    st.toast(f"Deleted {f}")
                    st.rerun()
    else:
        st.caption("No documents in knowledge base.")

    st.divider()
    # Metadata filter dropdown
    selected_filter = st.selectbox(
        "Search in specific document", 
        ["All Documents"] + doc_files, 
        help="Limits semantic search to a specific document."
    )
    filename_filter = selected_filter if selected_filter != "All Documents" else None

# ==========================================
# Main Workspace: Chat Interface
# ==========================================
st.title(f"💬 DocQuery - Workspace: {st.session_state.user_id}")
st.caption("Ask questions about your documents in real-time.")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if they exist for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander(f"View Sources ({message['num_sources_used']})"):
                for idx, src in enumerate(message["sources"], 1):
                    meta = src.metadata if hasattr(src, "metadata") else {}
                    st.markdown(f"**Source {idx}:** {meta.get('filename', 'Unknown')} (Page {meta.get('page_number', 'N/A')})")
                    st.caption(f"_{src.page_content}_")

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        # 1. Retrieve Docs First before opening stream
        with st.spinner(f"Retrieving from {selected_filter}..."):
            docs = st.session_state.retrieval_mgr.retrieve(prompt, filename_filter=filename_filter)
        
        if not docs:
            st.warning("No relevant sources found for this query.")
            st.session_state.messages.append({"role": "assistant", "content": "No relevant sources found.", "sources": [], "num_sources_used": 0})
        else:
            try:
                # 2. Generate w/ Stream
                stream, sources = st.session_state.generator.generate_stream(prompt, docs)
                
                # 3. Stream Output
                answer = st.write_stream(stream)
                
                # Display Sources Accordion
                with st.expander(f"View Sources ({len(docs)})"):
                    for idx, src in enumerate(docs, 1):
                        meta = src.metadata if hasattr(src, "metadata") else {}
                        st.markdown(f"**Source {idx}:** {meta.get('filename', 'Unknown')} (Page {meta.get('page_number', 'N/A')})")
                        st.caption(f"_{src.page_content}_")
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": docs,
                    "num_sources_used": len(docs)
                })
                
            except Exception as e:
                error_msg = f"Sorry, an error occurred while streaming response.\nError: `{e}`"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
