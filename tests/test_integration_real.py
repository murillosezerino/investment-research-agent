"""Integration tests that exercise a REAL ChromaDB, with no mocks and no OpenAI.

Deterministic keyword-based embeddings are injected so retrieval is meaningful
while staying fully offline and reproducible. This proves the RAG and memory
layers work end to end against a real vector store, not only against mocks.
"""

from langchain_core.embeddings import Embeddings

import src.memory.store as memory_store
import src.rag.ingestion as ingestion
import src.rag.retriever as retriever_mod
from src.config import settings
from src.memory.models import MemoryRecord
from src.memory.store import MemoryStore
from src.rag.ingestion import ingest_documents
from src.rag.retriever import get_retriever


class KeywordEmbeddings(Embeddings):
    """Offline, deterministic embeddings: normalized keyword-count vectors.

    Text that shares keywords maps to nearby vectors, so nearest-neighbor
    search is meaningful without any external embedding API.
    """

    VOCAB = [
        "imobili", "fii", "vacanc", "vacân", "dividend",
        "tesouro", "ipca", "inflac", "inflaç", "marcac",
        "etf", "bova", "ibovespa", "indice", "índice", "diversif",
    ]

    def _vec(self, text: str) -> list[float]:
        t = text.lower()
        v = [float(t.count(w)) for w in self.VOCAB]
        norm = sum(x * x for x in v) ** 0.5
        return [x / norm for x in v] if norm else [1.0] + [0.0] * (len(self.VOCAB) - 1)

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def test_memory_roundtrip_real_chroma(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "memory_persist_dir", str(tmp_path / "mem"))
    monkeypatch.setattr(settings, "memory_collection", "test_memory")
    monkeypatch.setattr(memory_store, "get_embeddings", lambda: KeywordEmbeddings())

    mem = MemoryStore()
    assert mem.count() == 0  # exercises the public-API count(), no Chroma internals

    mem.remember(MemoryRecord(
        question="Vale a pena investir em fundos imobiliários (FII)?",
        recommendation="FIIs pagam dividendos mensais; risco de vacância.",
    ))
    mem.remember(MemoryRecord(
        question="Como funciona o Tesouro IPCA+?",
        recommendation="Título indexado à inflação, com marcação a mercado.",
    ))
    assert mem.count() == 2

    recalled = mem.recall("risco de vacância em fundos imobiliários", k=1)
    assert len(recalled) == 1
    assert "imobili" in recalled[0].question.lower()


def test_rag_ingest_and_retrieve_real_chroma(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "corpus"))
    monkeypatch.setattr(settings, "collection_name", "test_corpus")
    monkeypatch.setattr(ingestion, "get_embeddings", lambda: KeywordEmbeddings())
    monkeypatch.setattr(retriever_mod, "get_embeddings", lambda: KeywordEmbeddings())

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "fii.txt").write_text(
        "Fundos imobiliários FII pagam dividendos; risco de vacância dos imóveis.",
        encoding="utf-8",
    )
    (docs_dir / "tesouro.txt").write_text(
        "Tesouro IPCA+ é indexado à inflação, com marcação a mercado.",
        encoding="utf-8",
    )
    (docs_dir / "etf.txt").write_text(
        "ETF BOVA11 replica o Ibovespa e serve para diversificação passiva.",
        encoding="utf-8",
    )

    ingest_documents(docs_dir)

    retriever = get_retriever(k=1)
    hits = retriever.invoke("vacância em fundos imobiliários")
    assert len(hits) == 1
    assert "imobili" in hits[0].page_content.lower()
