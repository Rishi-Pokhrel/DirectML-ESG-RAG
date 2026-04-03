import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import logging
from src.utils.logger import setup_logger

logger = setup_logger("vector_store")

class VectorStore:
    """Handles vector embedding and storage using ChromaDB and SentenceTransformers."""
    
    def __init__(self, config_path: str = "config/settings.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        self.embedding_model = SentenceTransformer(
            self.config["embedding"]["model_name"],
            device=self.config["embedding"]["device"]
        )
        
        self.client = chromadb.PersistentClient(path=self.config["rag"]["vector_db_path"])
        self.collection = self.client.get_or_create_collection(
            name=self.config["rag"]["collection_name"],
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Initialized VectorStore at {self.config['rag']['vector_db_path']}")

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """Embeds and adds documents to the collection."""
        logger.info(f"Adding {len(texts)} chunks to vector store...")
        embeddings = self.embedding_model.encode(texts).tolist()
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        logger.info("Successfully added documents.")

    def query(self, query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Searches for the most similar chunks."""
        query_embedding = self.embedding_model.encode([query_text]).tolist()[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format results for readability
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })
        return formatted_results

    def get_count(self) -> int:
        """Returns total number of documents in collection."""
        return self.collection.count()
