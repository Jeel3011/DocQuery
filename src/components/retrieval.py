import os
import sys 
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from src.components.config import Config
from src.logger import get_logger
logger = get_logger(__name__)

class RetrievalManager:
    def __init__(self,config:Config):
        self.config = config
        self.logger = logger

        self.vectorstore = Chroma(
            persist_directory=self.config.VECTOR_DB_PATH,
            embedding_function=OpenAIEmbeddings(model=self.config.EMBEDDING_MODEL_NAME),
            collection_name=self.config.COLLECTION_NAME,
            collection_metadata={"hnsw:space":"cosine"}
        )

        self.retriever =self.vectorstore.as_retriever(
            search_type = "mmr",
            search_kwargs={
                "k":self.config.TOP_K,
                "fetch_k":15,
                "lambda_mult":0.5
            }
        ) 

       

    def retriev(self,query):

        docs=self.retriever.invoke(query)
        return docs


if __name__ == "__main__" :
    config = Config()
    retrieval_manager = RetrievalManager(config=config)
    query = "what is Decoder?"
    docs = retrieval_manager.retriev(query) 
    print(docs)  