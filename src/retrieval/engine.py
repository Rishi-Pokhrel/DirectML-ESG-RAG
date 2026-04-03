import logging
import gc
from typing import List, Dict, Any
from src.retrieval.vector_store import VectorStore
from src.models.local_llm import LocalLLM
from src.utils.logger import setup_logger

logger = setup_logger("retrieval_engine")

class RAGEngine:
    """Precision RAG Engine: Uses Micro-Chunk retrieval for high-density technical context."""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.local_llm = LocalLLM()
        
    def answer_query(self, query: str) -> Dict[str, Any]:
        """Performs Precision RAG to answer user query with exhaustive technical detail."""
        logger.info(f"Precision Querying RAG: {query}")
        
        # 1. Retrieve more (but smaller) chunks to increase context density
        results = self.vector_store.query(query, n_results=10)
        context = "\n---\n".join([r["text"] for r in results])
        
        # 2. Optimized Technical Prompt for 0.5B Model
        prompt = f"""You are an Expert Automotive Service Assistant. Using ONLY the technical snippets below, provide a COMPLETE, EXHAUSTIVE, STEP-BY-STEP answer. 

RULES:
- If troubleshooting, list every diagnostic step found in context.
- Maintain a professional technical tone.
- Do not summarize if it removes technical details (e.g., torques, specific checks).
- If the answer isn't in context, state: 'Information not found in available technical manuals.'

TECHNICAL CONTEXT SNIPPETS:
{context}

USER QUESTION: 
{query}

EXHAUSTIVE TECHNICAL ANSWER:"""
        
        # 3. Generate answer
        answer = self.local_llm.generate(prompt)
        
        # 4. Explicit Memory Cleanup for 520MB Target
        gc.collect()
        
        # 5. Prepare response
        response = {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "file": r["metadata"]["source"],
                    "page": r["metadata"]["page"],
                    "type": r["metadata"]["type"]
                } for r in results
            ]
        }
        logger.info("Exhaustive Answer generated successfully.")
        return response
