from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # where to store FAISS files (relative to rag-service/)
    index_path: str = "data/index.faiss"

    class Config:
        env_file = "../../.env"  # will create this later

settings = Settings()
