import json
import os
from typing import List, Dict, Any

from rag_engine.embedding.embedder import LocalEmbedder
from rag_engine.vector_store.repository import VectorRepository

class RAGAgent:
    def __init__(self):
        print("[*] Initializing RAG Agent...")
        self.embedder = LocalEmbedder()
        self.db = VectorRepository()

    def ingest_data(self, json_path: str):        
        """
        Reads chunked data from rag_engine/data/ file, embeds the text, and stores in the ChromaDB.
        """

        if json_path is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            json_path = os.path.join(base_dir, "data", "sample_chunks.json")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"[-] The file {json_path} does not exist.")

        print(f"[*] Reading data from {json_path}...")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)

        ids: List[str] = []
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks_data:
            # IDs should to be strings
            ids.append(str(chunk["id"]))
            texts.append(chunk["text"])
            # Storing context_id as metadata
            metadatas.append({"context_id": chunk["context_id"]})

        print(f"[+] Successfully loaded {len(texts)} chunks from JSON.")


        print("[*] Generating embeddings for the chunks...")
        embeddings = self.embedder.embed_documents(texts)
        print("[+] Embeddings generated successfully.")

        print("[*] Storing data in ChromaDB...")
        self.db.add_chunks(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        print("[+] Data ingestion complete! DB is ready.")

    def ask_question(self, question: str, top_k: int = 3):
        print(f"[*] Processing question: '{question}'")
        
        question_embedding = self.embedder.embed_query(question)
        
        print(f"[*] Searching vector store for top {top_k} most relevant chunks...")
        search_results = self.db.search_similar(
            query_embedding=question_embedding, 
            n_results=top_k
        )
        
        retrieved_documents = search_results.get("documents", [[]])[0]
        retrieved_metadatas = search_results.get("metadatas", [[]])[0]
        retrieved_ids = search_results.get("ids", [[]])[0]
        retrieved_distances = search_results.get("distances", [[]])[0] if search_results.get("distances") else []
        
        print("[+] Retrieval successful. Retrieved contexts:")
        
        formatted_chunks = []
        
        for i in range(len(retrieved_documents)):
            distance = retrieved_distances[i] if i < len(retrieved_distances) else None
            
            similarity_percentage = None
            if distance is not None:
                similarity_percentage = round((1 - distance) * 100, 2)
            
            chunk_data = {
                "id": retrieved_ids[i] if i < len(retrieved_ids) else None,
                "metadata": retrieved_metadatas[i] if i < len(retrieved_metadatas) else {},
                "text": retrieved_documents[i],
                "distance": distance,
                "similarity_percentage": similarity_percentage
            }
            formatted_chunks.append(chunk_data)
            
            print(f"\n--- Chunk {i+1} ---")
            print(f"ID: {chunk_data['id']}")
            
            if chunk_data['similarity_percentage'] is not None:
                print(f"Similarity: {chunk_data['similarity_percentage']}%  (Raw Distance: {chunk_data['distance']:.4f})")
                
            print("Metadata:")
            for key, value in chunk_data['metadata'].items():
                print(f"  - {key}: {value}")
                
            print(f"Text:\n{chunk_data['text']}")
            
        return formatted_chunks