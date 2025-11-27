from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    CHROMA_DB_PATH: str = "./data/rag_db"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    RERANK_MODEL_NAME: str = "hotchpotch/japanese-bge-reranker-v2-m3-v1"
    RERANK_MODEL_CACHE: str = "./data/model"

    class Config:
        env_file = ".env"

settings = Settings()
