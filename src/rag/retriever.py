from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import settings
from src.rag.embeddings import get_embeddings


def get_retriever(k: int = 4) -> VectorStoreRetriever:
    """Return a retriever backed by the persisted ChromaDB vector store."""
    vector_store = Chroma(
        persist_directory=settings.chroma_persist_dir,
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
    )
    return vector_store.as_retriever(search_kwargs={"k": k})
