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

    def rewrite_query(self, query: str, chat_history: list = None) -> str:
        """
        Rewrite a follow-up question to be a standalone query based on chat history.
        This ensures the vector DB can find relevant documents even for ambiguous queries like 'what is it?'.
        """
        if not chat_history:
            return query
            
        # format_chat_history already handles truncation
        formatted_history = format_chat_history(chat_history)
        if not formatted_history:
            return query
            
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that rewrites conversational follow-up questions into standalone search queries.
Given the conversation history, rephrase the follow-up question to be a standalone query that can be used to search a document database. 
- Resolve any pronouns (it, they, them, this, these, those) to their specific referents from the history.
- If the user uses vague terms like "all 3" or "both", explicitly list what those refer to based on the previous messages.
- Do not answer the question, ONLY return the rewritten standalone question.
- If the question is already completely standalone, return it exactly as is.

Conversation History:
{chat_history}"""),
            ("user", "Follow-up question: {question}\nStandalone query:")
        ])
        
        chain = rewrite_prompt | self.llm | StrOutputParser()
        standalone_query = chain.invoke({
            "chat_history": formatted_history,
            "question": query
        })
        
        logger.info(f"Rewrote query: '{query}' -> '{standalone_query}'")
        return standalone_query.strip()

    def generate(self,query:str,retrieved_docs: List[Document], chat_history: list = None) -> Dict:

            
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
                "question":query,
                "chat_history": format_chat_history(chat_history) if chat_history else ""
            })

            return {"answer":answer,"sources":sources,"num_sources_used":len(retrieved_docs)}
    
    def generate_stream(self, query: str, retrieved_docs: List[Document], chat_history: list = None):
        """Yield Server-Sent Events (SSE) for streaming RAG responses."""
        context_parts = []
        sources = []
        for i, doc in enumerate(retrieved_docs, 1):
            meta = doc.metadata
            source_info = f"[Source{i}: {meta.get('filename','unknown')}, Page: {meta.get('page_number', 'N/A')} , Type: {meta.get('chunk_type', 'text')} ]"
            context_parts.append(f"{source_info}\n{doc.page_content}\n")
            
            # Prepare source metadata for the client, including a snippet of the text
            snippet = doc.page_content[:250].replace('\n', ' ') + "..."
            sources.append({
                "source_id": i,
                "filename": meta.get('filename', 'Unknown'),
                "page": meta.get('page_number', 'N/A'),
                "chunk_type": meta.get('chunk_type', 'text'),
                "chunk_id": meta.get('chunk_id', ''),
                "content": snippet
            })

        context = "\n---\n".join(context_parts)
        
        import json
        
        # 1. Send the sources to the client first
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        
        # 2. Stream the LLM tokens
        stream = self.chain.stream({
            "context": context,
            "question": query,
            "chat_history": format_chat_history(chat_history) if chat_history else ""
        })
        
        for chunk in stream:
            # Depending on OutputParser, chunk might be a string or dict
            text_chunk = chunk if isinstance(chunk, str) else chunk.get("answer", "")
            if text_chunk:
                yield f"data: {json.dumps({'type': 'token', 'content': text_chunk})}\n\n"
                
        # 3. Signal end of stream
        yield "data: [DONE]\n\n"


            

if __name__ == "__main__" :
    config = Config()
    retrieval_manager = RetrievalManager(config=config)
    generator = AnswerGenration(config=config)
    query = "expalin Multi-Head Attention"
    docs = retrieval_manager.retrieve(query) 
    results = generator.generate(query,docs)
    print(results["answer"])              