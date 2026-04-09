from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.rag.ingestion import _load_documents, _split_documents


class TestLoadDocuments:
    def test_loads_txt_files(self, tmp_docs_dir: Path):
        docs = _load_documents(tmp_docs_dir)
        assert len(docs) >= 1
        assert any("12.3%" in doc.page_content for doc in docs)

    def test_raises_on_empty_dir(self, tmp_path: Path):
        with pytest.raises(ValueError, match="No supported documents"):
            from src.rag.ingestion import ingest_documents

            with patch("src.rag.ingestion.get_embeddings", return_value=MagicMock()):
                ingest_documents(tmp_path)

    def test_ignores_unsupported_files(self, tmp_path: Path):
        (tmp_path / "data.csv").write_text("a,b,c", encoding="utf-8")
        docs = _load_documents(tmp_path)
        assert len(docs) == 0


class TestSplitDocuments:
    def test_splits_long_document(self):
        long_text = "Investimento em renda fixa. " * 200
        docs = [Document(page_content=long_text, metadata={"source": "test"})]
        chunks = _split_documents(docs)
        assert len(chunks) > 1

    def test_preserves_metadata(self):
        docs = [Document(page_content="Short text.", metadata={"source": "test.pdf", "page": 5})]
        chunks = _split_documents(docs)
        assert all(c.metadata["source"] == "test.pdf" for c in chunks)

    def test_short_document_stays_single_chunk(self):
        docs = [Document(page_content="Texto curto.", metadata={"source": "test"})]
        chunks = _split_documents(docs)
        assert len(chunks) == 1
