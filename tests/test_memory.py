"""Tests for the long-term memory layer."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.memory import MemoryRecord, format_memories


class TestMemoryRecord:
    def test_to_metadata_drops_empty_values(self):
        record = MemoryRecord(question="Risco de FIIs?", recommendation="HOLD")
        meta = record.to_metadata()
        assert meta["recommendation"] == "HOLD"
        assert "risk_score" not in meta  # empty values are dropped
        assert "timestamp" not in meta

    def test_to_metadata_keeps_provided_values(self):
        record = MemoryRecord(
            question="Risco de FIIs?",
            recommendation="HOLD",
            risk_score="5",
            timestamp="2025-01-01T00:00:00+00:00",
        )
        meta = record.to_metadata()
        assert meta["risk_score"] == "5"
        assert meta["timestamp"] == "2025-01-01T00:00:00+00:00"

    def test_from_document_roundtrip(self):
        doc = Document(
            page_content="Risco de FIIs?",
            metadata={"recommendation": "HOLD", "risk_score": "5"},
        )
        record = MemoryRecord.from_document(doc)
        assert record.question == "Risco de FIIs?"
        assert record.recommendation == "HOLD"
        assert record.risk_score == "5"
        assert record.timestamp is None


class TestFormatMemories:
    def test_empty_returns_placeholder(self):
        assert "No prior related research" in format_memories([])

    def test_renders_records(self):
        records = [
            MemoryRecord(question="Risco de FIIs?", recommendation="HOLD", risk_score="5"),
            MemoryRecord(question="ETFs valem a pena?", recommendation="BUY"),
        ]
        out = format_memories(records)
        assert "Risco de FIIs?" in out
        assert "HOLD" in out
        assert "risk score: 5" in out
        assert "ETFs valem a pena?" in out


class TestMemoryStore:
    def _build_store(self, chroma_mock):
        with patch("src.memory.store.Chroma", return_value=chroma_mock), patch(
            "src.memory.store.get_embeddings", return_value=MagicMock()
        ):
            from src.memory.store import MemoryStore

            return MemoryStore()

    def test_remember_persists_and_returns_id(self):
        chroma_mock = MagicMock()
        chroma_mock.add_texts.return_value = ["mem-123"]
        store = self._build_store(chroma_mock)

        memory_id = store.remember(MemoryRecord(question="Risco de FIIs?", recommendation="HOLD"))

        assert memory_id == "mem-123"
        # question is what gets embedded; recommendation rides in metadata
        args, kwargs = chroma_mock.add_texts.call_args
        assert kwargs["texts"] == ["Risco de FIIs?"]
        assert kwargs["metadatas"][0]["recommendation"] == "HOLD"
        # a timestamp is stamped automatically when absent
        assert kwargs["metadatas"][0]["timestamp"]

    def test_recall_maps_documents_to_records(self):
        chroma_mock = MagicMock()
        chroma_mock.similarity_search.return_value = [
            Document(page_content="Risco de FIIs?", metadata={"recommendation": "HOLD"}),
        ]
        store = self._build_store(chroma_mock)

        records = store.recall("fundos imobiliários", k=2)

        chroma_mock.similarity_search.assert_called_once_with("fundos imobiliários", k=2)
        assert len(records) == 1
        assert records[0].recommendation == "HOLD"

    def test_recall_with_zero_k_short_circuits(self):
        chroma_mock = MagicMock()
        store = self._build_store(chroma_mock)

        assert store.recall("anything", k=0) == []
        chroma_mock.similarity_search.assert_not_called()
