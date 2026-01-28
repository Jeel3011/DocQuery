
from typing import List,Dict, Any
from src.components.config import Config
from src.components.data_ingestion import DocumentProcessor
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

class EmbeddingManager:
    def __init__(self,config : Config):
        self.config = config

    def create_vector_store(self,documents : List[Document],persist_directory: str) -> Chroma:


        try:
            embedding_model = OpenAIEmbeddings(model=self.config.EMBEDDING_MODEL_NAME,
                                                openai_api_key=self.config.OPENAI_API_KEY)
            
            vector_store = Chroma.from_documents(documents=documents,
                                                 embedding=embedding_model,
                                                 persist_directory=persist_directory,
                                                 collection_configuration={"hnsw:space":"cosine"}
                                            )
            
            print("Vector store created successfully.")
            return vector_store
        except Exception as e:
            print(f"Error creating vector store: {e}")


if __name__ == "__main__":
    config = Config()  
    doc_processor= DocumentProcessor(config=config)
    documents = doc_processor.process_batch(directory=config.UPLOAD_DIR)
    embed = EmbeddingManager(config=config)
    Vector_store = embed.create_vector_store(documents=documents,persist_directory=config.VECTOR_DB_PATH)



       


