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

class EmbeddingManager:
    def __init__(self,config : Config):
        self.config = config
        self.logger = logger
        if self.config.PINECONE_API_KEY:
            os.environ["PINECONE_API_KEY"] = self.config.PINECONE_API_KEY

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
            

    def create_vector_store(self,documents : List[Document],persist_directory: str = None) -> PineconeVectorStore:
        embedding_model = OpenAIEmbeddings(
            model=self.config.EMBEDDING_MODEL_NAME,
            openai_api_key=self.config.OPENAI_API_KEY
        )
        
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
            
            # Upsert into Pinecone. This automatically handles duplicates if reusing chunk_id.
            vector_store.add_documents(documents=documents, ids=[d.metadata["chunk_id"] for d in documents])
            
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




