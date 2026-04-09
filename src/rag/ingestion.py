from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.rag.embeddings import get_embeddings

LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}


def _load_documents(source_dir: str | Path) -> list[Document]:
    """Load documents from a directory, supporting PDF, TXT and MD files."""
    source_path = Path(source_dir)
    documents: list[Document] = []

    for file_path in source_path.iterdir():
        loader_cls = LOADERS.get(file_path.suffix.lower())
        if loader_cls is None:
            continue
        loader = loader_cls(str(file_path))
        documents.extend(loader.load())

    return documents


def _split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def ingest_documents(source_dir: str | Path) -> Chroma:
    """Load, split and index documents into ChromaDB vector store."""
    documents = _load_documents(source_dir)
    if not documents:
        raise ValueError(f"No supported documents found in {source_dir}")

    chunks = _split_documents(documents)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
        collection_name=settings.collection_name,
    )
    return vector_store
