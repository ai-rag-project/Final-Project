from sentence_transformers import SentenceTransformer
from typing import List

from rag_engine.config.config import settings

class LocalEmbedder:
    def __init__(self):
        
        self.model_name = settings.EMBEDDING_MODEL_NAME
        print(f"[*] Loading embedding model: {self.model_name}...")
        print("[*] If this is the first run, it might take a minute to download (~436MB).")
        
        # Load or download model
        self.model = SentenceTransformer(self.model_name)
        print("[+] Model loaded successfully!")

    def embed_query(self, text: str) -> List[float]:
        
        instruction = "Represent this sentence for searching relevant passages: "
        formatted_text = instruction + text
        
        return self.model.encode(formatted_text).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def get_dimension(self) -> int:
        return settings.EMBEDDING_DIMENSION