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
            embedding_function=OpenAIEmbeddings(
                model=self.config.EMBEDDING_MODEL_NAME,
                openai_api_key=self.config.OPENAI_API_KEY,
            ),
            collection_name=self.config.COLLECTION_NAME,
            collection_metadata={"hnsw:space": "cosine"}
        )

    def retrieve(self, query: str, filename_filter: str = None, page_filter: str = None):
        """Retrieve relevant docs, filtered by SIMILARITY_THRESHOLD (cosine distance)."""
        # cosine distance threshold: distance = 1 - similarity
        distance_threshold = 1.0 - self.config.SIMILARITY_THRESHOLD
        try:
            if filename_filter:
                filter_dict = {'filename': filename_filter}
                if page_filter:
                    filter_dict['page_number'] = page_filter
                docs_and_scores = self.vectorstore.similarity_search_with_score(
                    query, k=self.config.TOP_K, filter=filter_dict
                )
            else:
                docs_and_scores = self.vectorstore.similarity_search_with_score(
                    query, k=self.config.TOP_K
                )
            # Filter out low-quality results below the similarity threshold
            docs = [doc for doc, score in docs_and_scores if score <= distance_threshold]
            self.logger.info(f"Retrieved {len(docs)}/{len(docs_and_scores)} docs above threshold")
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

