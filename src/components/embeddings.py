from typing import List
from src.components.config import Config
from src.components.data_ingestion import DocumentProcessor
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import hashlib
import traceback
from src.logger import get_logger
logger = get_logger(__name__)

class EmbeddingManager:
    def __init__(self,config : Config):
        self.config = config
        self.logger = logger

    @staticmethod
    def hash_content(text:str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
            

    def create_vector_store(self,documents : List[Document],persist_directory: str) -> Chroma:
        self.logger.info(
            "starting vector creation",
            extra={
                "num_documents" : len(documents),
                "persist_directory": persist_directory,
                "embeding_model" : self.config.EMBEDDING_MODEL_NAME,
            }
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

            embedding_model = OpenAIEmbeddings(model=self.config.EMBEDDING_MODEL_NAME,
                                                openai_api_key=self.config.OPENAI_API_KEY)

            self.logger.info("Embedding model created")                                    

            for doc in documents:
                doc.metadata["chunk_id"] = f"{doc.metadata.get('source','unknown')}::{doc.metadata['content_hash']}"


            vector_store =Chroma(
                                    embedding_function=embedding_model,

                                    persist_directory=persist_directory,

                                    collection_metadata={"hnsw:space":"cosine"},
                                    collection_name=self.config.COLLECTION_NAME
                                            )
            
            ids = [doc.metadata["chunk_id"] for doc in documents]

            vector_store.add_documents(
                documents=documents,
                ids = ids
            )
            

            print("Vector store created successfully.")

            
            
            return vector_store
        
        except Exception as e:
            self.logger.exception(
                "Failed to create vector store",
                extra={
                    "persist_directory": persist_directory,
                    "num_documents": len(documents),
                },
            )
            raise
           


if __name__ == "__main__":
    config = Config()  
    doc_processor= DocumentProcessor(config=config)
    documents = doc_processor.process_batch(directory=config.UPLOAD_DIR)
    embed = EmbeddingManager(config=config)
    Vector_store = embed.create_vector_store(documents=documents,persist_directory=config.VECTOR_DB_PATH)



       


