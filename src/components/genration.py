from typing import List,Dict
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.components.retrieval import RetrievalManager
from src.components.config import Config
from src.logger import get_logger
logger = get_logger(__name__)
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
from src.utils import format_chat_history


class AnswerGenration():
    def __init__(self,config:Config):
        self.config = config
        self.llm = ChatOpenAI(
            model=self.config.LLM_MODEL_NAME,
            temperature=0.1,
            api_key=self.config.OPENAI_API_KEY
        )

        self.prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant answering questions based on provided documents.


Rules:
1. Only use information from the context below
2. If answer is not in context, say "I cannot find information about your question from the document"
3. Cite sources using [Source: filename, Page: X] format
4. Be concise but complete
5. Use conversation history only to understand follow-up questions, not as a source of facts

Conversation History:
{chat_history}

Context:
{context}
"""),
    ("user", "{question}")
])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate(self,query:str,retrieved_docs: List[Document]) -> Dict:

            
            context_parts=[]
            sources = []

            for i,doc in enumerate(retrieved_docs,1):
                meta = doc.metadata
                source_info = f"[Source{i}: {meta.get('filename','unknown')}, Page: {meta.get('page_number', 'N/A')}, Type: {meta.get('chunk_type', 'text')} ]"
                context_parts.append(f"{source_info}\n{doc.page_content}\n")

                sources.append({
                    "source_id": i,
                    "filename":meta.get('filename'),
                    "page":meta.get('page_number'),
                    "chunk_type":meta.get('chunk_type'),
                    "chunk_id":meta.get('chunk_id')
                })
                                     
            context = "\n---\n".join(context_parts)


            answer = self.chain.invoke({
                "context":context,
                "question":query
            })

            return {"answer":answer,"sources":sources,"num_sources_used":len(retrieved_docs)}
    
    def generate_stream(self, query: str, retrieved_docs: List[Document],chat_history:list = None):
        context_parts = []
        sources = []
        for i, doc in enumerate(retrieved_docs, 1):
            meta = doc.metadata
            source_info = f"[Source{i}: {meta.get('filename','unknown')}, Page: {meta.get('page_number', 'N/A')} , Type: {meta.get('chunk_type', 'text')} ]"
            context_parts.append(f"{source_info}\n{doc.page_content}\n")
            sources.append({
                "source_id": i,
                "filename": meta.get('filename'),
                "page": meta.get('page_number'),
                "chunk_type": meta.get('chunk_type'),
                "chunk_id": meta.get('chunk_id')})

        context = "\n---\n".join(context_parts)
    
        stream = self.chain.stream({"context": context,
                                    "question": query,
                                    "chat_history": format_chat_history(chat_history) if chat_history else ""
                                    })
        return stream, sources 


            

if __name__ == "__main__" :
    config = Config()
    retrieval_manager = RetrievalManager(config=config)
    generator = AnswerGenration(config=config)
    query = "expalin Multi-Head Attention"
    docs = retrieval_manager.retrieve(query) 
    results = generator.generate(query,docs)
    print(results["answer"])              