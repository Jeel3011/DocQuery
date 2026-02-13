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

class AnswerGenration():
    def __init__(self,config:Config):
        self.config = Config
        self.llm = ChatOpenAI(
            model=self.config.LLM_MODEL_NAME,
            temperature=0.1,
            api_key=self.config.OPENAI_API_KEY
        )

        self.prompt=ChatPromptTemplate.from_messages([
            ("system","""you are helpful assistant answering question based on provideddocuments.
             
             Rules: 
             1. Only use information from context below
             2. If answer is not in context, say " i can not find information about your question from document"
             3.Cite sources using[Sources :filename,Page : X] format
             4. Be conise but complete.

            Context : {context}

             """),
             ("user","{question}")

        ])

        self.chain = self.prompt | self.llm | StrOutputParser()


    def format_context(self,retrieved_docs: List[Document]) -> tuple[str,list]:
        context_parts=[]
        sources = []
        for i,doc in enumerate(retrieved_docs,1):
                meta = doc.metadata
                source_info = f"[Source{1}: {meta.get('filename','unkown')}, Page: {meta.get('page_number', 'N/A')}, Type: {meta.get('chunk_type', 'text')} ]"
                context_parts.append(f"{source_info}\n{doc.page_content}\n")

                sources.append({
                    "source_id": i,
                    "filename":meta.get('filename'),
                    "page":meta.get('page_number'),
                    "chunk_type":meta.get('chunk_type'),
                    "chunk_id":meta.get('chunk_id')
                })
                                     
                context ="\n---\n.".join(context_parts) 
    def generate(self,query:str,retrieved_docs: List[Document]) -> Dict:

            
            context_parts=[]
            sources = []

            for i,doc in enumerate(retrieved_docs,1):
                meta = doc.metadata
                source_info = f"[Source{1}: {meta.get('filename','unkown')}, Page: {meta.get('page_number', 'N/A')}, Type: {meta.get('chunk_type', 'text')} ]"
                context_parts.append(f"{source_info}\n{doc.page_content}\n")

                sources.append({
                    "source_id": i,
                    "filename":meta.get('filename'),
                    "page":meta.get('page_number'),
                    "chunk_type":meta.get('chunk_type'),
                    "chunk_id":meta.get('chunk_id')
                })
                                     
                context ="\n---\n.".join(context_parts)

                answer = self.chain.invoke({
                    "context":context,
                    "question":query
                })

                return {"answer":answer,"sources":sources,"num_sources_used":len(retrieved_docs)}
            

if __name__ == "__main__" :
    config = Config()
    retrieval_manager = RetrievalManager(config=config)
    generator = AnswerGenration(config=config)
    query = "what is Decoder?"
    docs = retrieval_manager.retrieve(query) 
    results = generator.generate(query,docs)
    print(results["answer"])              