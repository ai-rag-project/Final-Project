import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import torch

from rag_engine.config.config import settings

class LocalEmbedder:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL_NAME

        # If gpu is available else using cpu 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[+] Using dvice: {self.device}")
        
        print(f"[*] Loading embedding model: {self.model_name} on {self.device.upper()}...")
        print("[*] If this is the first run, it might take a minute to download.")
        
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print("[+] Model loaded successfully!")
        except Exception as e:
            print(f"[-] Error loading model: {e}")
            raise e

    def embed_query(self, text: str) -> List[float]:
        instruction = "Represent this sentence for searching relevant passages: "
        formatted_text = instruction + text
        return self.model.encode(formatted_text).tolist()

    def embed_documents(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        return self.model.encode(texts, batch_size=batch_size).tolist()


    # ==========================================
    # File Processing & Chunking Methods
    # ==========================================
    def chunk_text(self, text: str, chunk_size: int = 170, overlap: int = 40) -> List[str]:
        # Chunking text with overlap
        words = text.split()
        chunks = []
        
        if not words:
            return chunks
            
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
                
        return chunks

    def process_and_embed_file(self, file_path: str) -> Dict[str, List]:
        """
        return chunks and embedding of each one
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' does not exist.")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        print(f"[*] Reading and chunking file: {file_path}")
        chunks = self.chunk_text(content)
        print(f"[+] Generated {len(chunks)} chunks.")
        
        print("[*] Generating embeddings for chunks...")
        embeddings = self.embed_documents(chunks)
        print("[+] Embeddings generated successfully.")
        
        return {
            "chunks": chunks,
            "embeddings": embeddings
        }