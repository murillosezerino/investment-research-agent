"""MemoryStore — persistent, semantic long-term memory for the agent.

Each completed research session is written to a dedicated ChromaDB collection,
separate from the source-document collection used for RAG. Before answering a new
question the orchestrator recalls the most semantically similar past sessions and
replays them as context, so conclusions compound across runs instead of being
thrown away when the request ends.
"""

from datetime import UTC, datetime

from langchain_chroma import Chroma

from src.config import settings
from src.memory.models import MemoryRecord
from src.rag.embeddings import get_embeddings


class MemoryStore:
    """Vector-backed episodic memory persisted across sessions."""

    def __init__(self) -> None:
        self._store = Chroma(
            persist_directory=settings.memory_persist_dir,
            collection_name=settings.memory_collection,
            embedding_function=get_embeddings(),
        )

    def remember(self, record: MemoryRecord) -> str:
        """Persist a research interaction and return its memory id."""
        if record.timestamp is None:
            record.timestamp = datetime.now(UTC).isoformat()

        ids = self._store.add_texts(
            texts=[record.question],
            metadatas=[record.to_metadata()],
        )
        return ids[0]

    def recall(self, query: str, k: int | None = None) -> list[MemoryRecord]:
        """Return the past records most semantically similar to ``query``."""
        k = k if k is not None else settings.memory_recall_k
        if k <= 0:
            return []
        docs = self._store.similarity_search(query, k=k)
        return [MemoryRecord.from_document(doc) for doc in docs]

    def count(self) -> int:
        """Number of memories currently stored."""
        return self._store._collection.count()


def format_memories(records: list[MemoryRecord]) -> str:
    """Render recalled memories as a prompt-ready context block."""
    if not records:
        return "No prior related research found in memory."

    blocks = []
    for i, r in enumerate(records, start=1):
        risk = f" (risk score: {r.risk_score})" if r.risk_score else ""
        blocks.append(
            f"{i}. Past question: {r.question}{risk}\n"
            f"   Prior conclusion: {r.recommendation}"
        )
    return "\n".join(blocks)
