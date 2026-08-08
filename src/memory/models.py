"""Schemas for the long-term memory layer."""

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class MemoryRecord(BaseModel):
    """A single episodic memory: one past research interaction.

    The ``question`` is what gets embedded and searched semantically; the rest
    is metadata carried alongside so a recalled memory can be replayed as
    context for a new, related question.
    """

    question: str
    recommendation: str
    risk_score: str | None = None
    timestamp: str | None = None
    memory_id: str | None = Field(default=None, description="ChromaDB document id")

    def to_metadata(self) -> dict:
        """Flatten into a Chroma-compatible metadata dict (no None values)."""
        meta = {
            "recommendation": self.recommendation,
            "risk_score": self.risk_score or "",
            "timestamp": self.timestamp or "",
        }
        return {k: v for k, v in meta.items() if v != ""}

    @classmethod
    def from_document(cls, doc: Document) -> "MemoryRecord":
        """Rehydrate a record from a stored Chroma document."""
        meta = doc.metadata or {}
        return cls(
            question=doc.page_content,
            recommendation=meta.get("recommendation", ""),
            risk_score=meta.get("risk_score") or None,
            timestamp=meta.get("timestamp") or None,
            memory_id=meta.get("memory_id") or None,
        )
