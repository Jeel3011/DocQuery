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
            search_type = "similarity",

            search_kwargs={
                "k":self.config.TOP_K,
            }
        ) 

    def retrieve(self,query):
        try:
            docs=self.retriever.invoke(query)
            return docs
        except Exception as e:
            self.logger.error(f"retrieval failed : {e}")
            return []


