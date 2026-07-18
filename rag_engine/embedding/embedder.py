from typing import List
from sentence_transformers import SentenceTransformer
import torch

from rag_engine.config.config import settings

class LocalEmbedder:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL_NAME

        # If gpu is available else using cpu 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[+] Using device: {self.device}")
        
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