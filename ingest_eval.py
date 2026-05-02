from src.components.config import Config
from src.components.data_ingestion import DocumentProcessor
from src.components.embeddings import EmbeddingManager
import os
import shutil
import argparse
import sys
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

def main():
    parser = argparse.ArgumentParser(description="Ingest a single PDF into Pinecone eval namespace")
    parser.add_argument(
        "--pdf",
        type=str,
        default="tmp_uploads/attention_is_all_you_need.pdf",
        help="Path to the source PDF to ingest",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="eval_namespace",
        help="Pinecone namespace for evaluation data",
    )
    parser.add_argument(
        "--upload-dir",
        type=str,
        default="tmp_eval_dir",
        help="Temporary processing directory",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.pdf):
        print(f"[ERROR] PDF not found: {args.pdf}")
        sys.exit(1)

    config = Config()
    config.PINECONE_NAMESPACE = args.namespace
    config.UPLOAD_DIR = args.upload_dir
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)

    source_filename = os.path.basename(args.pdf)
    target_pdf_path = os.path.join(config.UPLOAD_DIR, source_filename)

    # copy pdf into processing directory
    shutil.copy2(
        args.pdf,
        target_pdf_path,
    )
    
    processor = DocumentProcessor(config)
    chunks = processor.process_batch(config.UPLOAD_DIR)
    
    print(f"Got {len(chunks)} chunks.")
    if not chunks:
        print("No chunks. Exiting.")
        return
        
    embed_mgr = EmbeddingManager(config)
    
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL_NAME,
        openai_api_key=config.OPENAI_API_KEY,
        request_timeout=30, # ensure no hang
        max_retries=3
    )
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    idx = pc.Index(config.PINECONE_INDEX_NAME)
    
    print("Embedding texts...")
    texts = [doc.page_content for doc in chunks]
    vectors = embeddings.embed_documents(texts)
    
    print("Upserting to Pinecone...")
    upsert_data = []
    for doc, vec in zip(chunks, vectors):
        doc.metadata["content_hash"] = embed_mgr.hash_content(doc.page_content)
        chunk_id = f"{doc.metadata.get('source','unknown')}::{doc.metadata['content_hash']}"
        doc.metadata["chunk_id"] = chunk_id
        meta = embed_mgr.clean_metadata(doc.metadata)
        meta["text"] = doc.page_content # Langchain pinecone needs this for retrieval
        upsert_data.append({"id": chunk_id, "values": vec, "metadata": meta})
    
    # upsert in batches of 100
    for i in range(0, len(upsert_data), 100):
        idx.upsert(vectors=upsert_data[i:i+100], namespace=config.PINECONE_NAMESPACE)
        
    print(f"Ingestion to namespace '{config.PINECONE_NAMESPACE}' complete!")

if __name__ == "__main__":
    main()
