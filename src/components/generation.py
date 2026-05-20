from typing import List, Dict
import json
import re
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
            api_key=self.config.OPENAI_API_KEY,
            request_timeout=30,   # P6: prevent infinite hang on OpenAI upstream stall
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

    def classify_query_complexity(self, query: str) -> str:
        """Classify query complexity without any LLM call — pure heuristics.

        Returns:
            "simple"   — Short or page-specific queries. Skip multi-query.
                         Use direct retrieval_by_vector for fastest path.
                         Expected latency: ~1.0-1.2s
            "moderate" — Multi-concept queries. Use multi-query retrieval.
                         Skip self-review. Expected latency: ~1.5-2.0s
            "complex"  — Comparative, analytical, multi-step. Full pipeline
                         including self-review. Expected latency: ~2.5-3.0s

        Harvey pattern: route cheap questions fast, reserve expensive LLM
        calls for genuinely hard questions.
        """
        q = query.lower().strip()
        words = q.split()

        # ── Simple signals ─────────────────────────────────────────────────
        # Page-specific queries — user wants a fact from a known location
        if re.search(r'\bpage\s+\d+\b', q):
            return "simple"

        # Very short queries — single concept lookup
        if len(words) <= 6:
            return "simple"

        # Definition / factual lookup starters
        if re.match(r'^(what is|what are|who is|who are|define|list|when|where|how many|how much)\b', q):
            return "simple"

        # ── Complex signals ────────────────────────────────────────────────
        # Comparative / analytical language
        complex_patterns = [
            r'\b(compare|contrast|difference between|similarities|versus|vs\.?)\b',
            r'\b(analyze|analyse|evaluate|assess|critique|explain why|explain how)\b',
            r'\b(relationship between|impact of|implications of|trade-?offs?)\b',
            r'\b(pros and cons|advantages and disadvantages)\b',
            r'\b(summarize|summarise|overview of|breakdown of)\b',
        ]
        for pattern in complex_patterns:
            if re.search(pattern, q):
                return "complex"

        # Long multi-clause queries are likely complex
        if len(words) > 25:
            return "complex"

        # Default: moderate — use multi-query but skip self-review
        return "moderate"

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

    def generate(self, query: str, retrieved_docs: List[Document], chat_history: list = None) -> Dict:

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

    def generate_with_self_review(
        self,
        query: str,
        retrieved_docs: List[Document],
        chat_history: list = None,
        user_id: str = None,
    ) -> Dict:
        """
        Phase 6: Harvey-style self-critique loop.

        Step 1 — Generate an initial answer normally.
        Step 2 — Ask the same LLM to identify any claims NOT directly supported
                  by the retrieved context. If all claims check out, respond VERIFIED.
                  If hallucinations are found, respond REVISE: [list].
        Step 3 — On REVISE, regenerate with an explicit strict-grounding instruction
                  that forbids adding information beyond the sources.

        Why this matters:
        - Harvey reports 0.2% error rate via citation-backed generation.
        - RAGAS faithfulness measures the same property. This loop directly
          addresses faithfulness failures on ambiguous or formula-heavy text.
        - Two LLM calls adds ~300-600ms latency — acceptable for the /query/agent
          endpoint (users expect it to be slower but more accurate).

        Returns the same dict as generate(), with extra fields:
            "self_reviewed": bool — whether the critique ran
            "revised":       bool — whether a revision was needed
            "critique":      str  — the raw critique output (for logging/debugging)
        """
        context_parts = []
        sources = []
        for i, doc in enumerate(retrieved_docs, 1):
            meta = doc.metadata
            source_info = f"[Source{i}: {meta.get('filename','unknown')}, Page: {meta.get('page_number', 'N/A')}, Type: {meta.get('chunk_type', 'text')} ]"
            context_parts.append(f"{source_info}\n{doc.page_content}\n")
            sources.append({
                "source_id": i,
                "filename": meta.get('filename'),
                "page": meta.get('page_number'),
                "chunk_type": meta.get('chunk_type'),
                "chunk_id": meta.get('chunk_id'),
            })
        context = "\n---\n".join(context_parts)

        # Step 1: Generate initial answer
        initial_answer = self.chain.invoke({
            "context": context,
            "question": query,
            "chat_history": format_chat_history(chat_history) if chat_history else "",
        })

        # Phase 4: token cost tracking (rough heuristic: len/4 ≈ tokens)
        if user_id:
            try:
                from src.components.metrics import user_llm_cost
                user_llm_cost.labels(
                    user_id=user_id,
                    model=self.config.LLM_MODEL_NAME,
                    operation="generate",
                ).inc(len(context) // 4)
            except Exception:
                pass

        # Step 2: Self-critique
        critique_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fact-checker reviewing an AI-generated answer.

The source documents contain:
{context}

The AI answered: "{answer}"

Identify any claims in the answer NOT directly supported by the sources above.
- If ALL claims are supported by the sources, respond exactly: VERIFIED
- If there are unsupported claims, respond: REVISE: [list the unsupported claims]

Respond with ONLY 'VERIFIED' or 'REVISE: ...' — nothing else."""),
            ("user", "Is the answer above fully supported by the source documents?"),
        ])
        critique_chain = critique_prompt | self.llm | StrOutputParser()

        critique = ""
        revised = False
        final_answer = initial_answer

        try:
            critique = critique_chain.invoke({
                "context": context,
                "answer": initial_answer,
            })

            if critique.strip().upper().startswith("REVISE"):
                logger.info("Self-review: revision needed. Critique: %s", critique[:120])
                revised = True

                # Step 3: Strict-grounding regeneration
                strict_prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a helpful assistant answering questions based ONLY on the provided documents.

CRITICAL: Do NOT add any information not explicitly present in the context below.
If the context does not contain enough information, say so explicitly.
Every claim must be directly traceable to a source document.

Context:
{context}

Conversation History:
{chat_history}"""),
                    ("user", "{question}"),
                ])
                strict_chain = strict_prompt | self.llm | StrOutputParser()
                final_answer = strict_chain.invoke({
                    "context": context,
                    "question": query,
                    "chat_history": format_chat_history(chat_history) if chat_history else "",
                })
            else:
                logger.info("Self-review: VERIFIED — no revision needed.")

        except Exception as exc:
            logger.warning("Self-review failed (non-fatal, using initial answer): %s", exc)

        return {
            "answer": final_answer,
            "sources": sources,
            "num_sources_used": len(retrieved_docs),
            "self_reviewed": True,
            "revised": revised,
            "critique": critique,
        }

    def generate_query_variants(self, query: str, n: int = 3) -> list[str]:
        """Generate N diverse search queries from the user's question for multi-query retrieval."""
        variant_prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate {n} different search queries to find relevant documents 
for the user's question. Each query should approach the topic from a different angle 
(e.g., synonyms, broader/narrower scope, related concepts).

Return ONLY the queries, one per line, no numbering or bullets."""),
            ("user", "{question}")
        ])
        chain = variant_prompt | self.llm | StrOutputParser()
        raw = chain.invoke({"question": query, "n": n})
        variants = [line.strip() for line in raw.strip().split("\n") if line.strip()]
        return variants[:n]

    
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
        
        # 2. Stream the LLM tokens (S7: wrap in try/except so client always gets [DONE])
        stream = self.chain.stream({
            "context": context,
            "question": query,
            "chat_history": format_chat_history(chat_history) if chat_history else ""
        })
        
        try:
            for chunk in stream:
                text_chunk = chunk if isinstance(chunk, str) else chunk.get("answer", "")
                if text_chunk:
                    yield f"data: {json.dumps({'type': 'token', 'content': text_chunk})}\n\n"
        except Exception as e:
            logger.error("SSE stream error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'message': 'Stream interrupted. Please try again.'})}\n\n"
        finally:
            # 3. Always signal end of stream so the client is never left hanging
            yield "data: [DONE]\n\n"


            

if __name__ == "__main__" :
    config = Config()
    retrieval_manager = RetrievalManager(config=config)
    generator = AnswerGenration(config=config)
    query = "expalin Multi-Head Attention"
    docs = retrieval_manager.retrieve(query) 
    results = generator.generate(query,docs)
    print(results["answer"])              