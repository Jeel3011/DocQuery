import time
from concurrent.futures import ThreadPoolExecutor
from typing import List
from src.components.config import Config
from src.components.data_ingestion import DocumentProcessor
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import hashlib
import traceback
import os
from src.logger import get_logger
logger = get_logger(__name__)

# A3: batch sizes for the embed/upsert stages.
#   EMBED_BATCH  — texts per OpenAI embed_documents() call; batches run in parallel.
#   UPSERT_BATCH — vectors per Pinecone upsert; upserts are issued asynchronously.
EMBED_BATCH = 256
UPSERT_BATCH = 64

class EmbeddingManager:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logger
        if self.config.PINECONE_API_KEY:
            os.environ["PINECONE_API_KEY"] = self.config.PINECONE_API_KEY

        # Initialize once — reused for all create_vector_store() calls.
        # Previously this was created fresh inside create_vector_store(), wasting
        # HTTP client setup on every ingestion task.
        self.embedding_model = OpenAIEmbeddings(
            model=self.config.EMBEDDING_MODEL_NAME,
            openai_api_key=self.config.OPENAI_API_KEY,
        )

    @staticmethod
    def hash_content(text:str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def clean_metadata(metadata: dict) -> dict:
        """Clean metadata by removing None values and converting invalid types to strings."""
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                continue  # Skip None values
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                # Convert non-standard types to string
                cleaned[key] = str(value)
        return cleaned
            

    def create_vector_store(self, documents: List[Document], persist_directory: str = None) -> PineconeVectorStore:
        # Use the embedding_model initialized in __init__ — no new HTTP client per call
        embedding_model = self.embedding_model

        # Early exit: if no documents, just return existing vector store
        if not documents:
            print("No new documents to embed. Using existing vector store.")
            return PineconeVectorStore(
                index_name=self.config.PINECONE_INDEX_NAME,
                embedding=embedding_model,
                namespace=self.config.PINECONE_NAMESPACE
            )

        try:
            # Deduplication
            unique_docs = []
            seen_hash = set()

            for doc in documents:
                content_hash= self.hash_content(doc.page_content)

                if content_hash in seen_hash:
                    continue

                seen_hash.add(content_hash)
                doc.metadata["content_hash"]=content_hash
                unique_docs.append(doc)

            documents=unique_docs    
            self.logger.info("Embedding model created")                                    

            for doc in documents:
                doc.metadata["chunk_id"] = f"{doc.metadata.get('source','unknown')}::{doc.metadata['content_hash']}"

            # Clean metadata before adding to vector store
            for doc in documents:
                doc.metadata = self.clean_metadata(doc.metadata)

            vector_store = PineconeVectorStore(
                index_name=self.config.PINECONE_INDEX_NAME,
                embedding=embedding_model,
                namespace=self.config.PINECONE_NAMESPACE
            )

            # A3: embed (parallel batches) then upsert (async batches), with an
            # explicit embed-vs-upsert timing breakdown. Behaviourally identical to
            # add_documents() — same chunk_id de-dupe, same "text"-key metadata —
            # but embedding batches run concurrently instead of one after another.
            self._embed_and_upsert(vector_store, documents)

            print("Vector store created successfully.")
            return vector_store
        
        except Exception as e:
            self.logger.exception(
                "Failed to create vector store",
                extra={
                    "num_documents": len(documents),
                },
            )
            raise

    def _embed_and_upsert(self, vector_store: PineconeVectorStore, documents: List[Document]) -> None:
        """A3: embed chunks in parallel batches, then async-upsert to Pinecone.

        Mirrors PineconeVectorStore.add_texts (full content stored under the
        "text" metadata key the retriever reads) but parallelizes the embedding
        batches and logs embed vs. upsert time separately.
        """
        if not documents:
            return

        # Match the retriever's text key so retrieved docs carry their content.
        text_key = getattr(vector_store, "_text_key", "text")
        texts = [d.page_content for d in documents]
        ids = [d.metadata["chunk_id"] for d in documents]
        metadatas = []
        for d in documents:
            md = dict(d.metadata)
            md[text_key] = d.page_content
            metadatas.append(md)

        # ── Embed: split into batches and run them concurrently ──
        t0 = time.perf_counter()
        batches = [texts[i:i + EMBED_BATCH] for i in range(0, len(texts), EMBED_BATCH)]
        if len(batches) <= 1:
            embeddings = self.embedding_model.embed_documents(texts)
        else:
            with ThreadPoolExecutor(max_workers=min(4, len(batches))) as ex:
                results = list(ex.map(self.embedding_model.embed_documents, batches))
            embeddings = [vec for batch in results for vec in batch]
        t_embed = time.perf_counter() - t0

        # ── Upsert: issue batched upserts asynchronously, then await them ──
        t1 = time.perf_counter()
        index = vector_store.index
        namespace = self.config.PINECONE_NAMESPACE
        vectors = list(zip(ids, embeddings, metadatas))
        async_results = [
            index.upsert(vectors=vectors[i:i + UPSERT_BATCH], namespace=namespace, async_req=True)
            for i in range(0, len(vectors), UPSERT_BATCH)
        ]
        for res in async_results:
            res.get()
        t_upsert = time.perf_counter() - t1

        self.logger.info(
            "A3 embed=%.2fs upsert=%.2fs (chunks=%d, embed_batches=%d)",
            t_embed, t_upsert, len(documents), len(batches),
        )




