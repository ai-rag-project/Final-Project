from pydantic_settings import BaseSettings

class RagSettings(BaseSettings):
    # General
    PROJECT_NAME: str = "RAG_Engine"
    
    # Embeding 
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIMENSION: int = 768
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = RagSettings()