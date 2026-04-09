from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    chroma_persist_dir: str = "./data/chroma"
    collection_name: str = "investments"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
