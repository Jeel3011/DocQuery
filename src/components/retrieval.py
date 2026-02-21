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

    def retrieve(self,query:str,filename_filter:str = None):

        try:
            if filename_filter:
                docs = self.vectorstore.similarity_search(query,k=self.config.TOP_K,filter={'filename':filename_filter})
            else:
                docs = self.retriever.invoke(query)

            return docs    
        except Exception as e:
            self.logger.error(f"retrieval failed : {e}")
            return []

    def delete_document_by_filename(self,filename:str):
        try:
            result = self.vectorstore.get(where={"filename":filename})
            ids = result.get('ids', [])
            if ids: 
                self.vectorstore.delete(ids=ids)
                self.logger.info(f"Deleted documents with filename {filename}")
            else:
                self.logger.warning(f"No documents found with filename {filename} to delete.")
        except Exception as e:
            self.logger.error(f"Failed to delete documents with filename {filename}: {e}")

