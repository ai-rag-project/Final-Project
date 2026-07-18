import chromadb
from typing import List, Dict, Any

from rag_engine.config.config import settings

class VectorRepository:
    def __init__(self, collection_name: str = "rag_collection", persist_directory: str = "./chroma_db"):
        print(f"[*] Initializing ChromaDB vector store at '{persist_directory}'...")
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} 
        )
        print(f"[+] Collection '{collection_name}' is ready.")

    def add_chunks(self, ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict[str, Any]]):
        print(f"[*] Adding {len(ids)} chunks to the vector store...")
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print("[+] Chunks added successfully!")

    def search_similar(self, query_embedding: List[float], n_results: int = 3) -> Dict[str, Any]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results

    def get_all_items(self) -> Dict[str, Any]:
        return self.collection.get(
            include=["embeddings", "documents", "metadatas"]
        )
    def count_items(self) -> int:
        return self.collection.count()
    
    def clear_collection(self):
        item_ids = self.collection.get()['ids']
        if item_ids:
            self.collection.delete(ids=item_ids)
            print("[+] Collection cleared.")