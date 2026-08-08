"""Tests for the MCP server tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory import MemoryRecord


class TestMcpServer:
    def test_server_registers_expected_tools(self):
        import asyncio

        from src.mcp_server.server import mcp

        tools = asyncio.run(mcp.list_tools())
        names = {t.name for t in tools}
        assert {
            "research_investment",
            "search_documents",
            "recall_memory",
            "remember_insight",
        } <= names


class TestMcpTools:
    @pytest.mark.asyncio
    async def test_research_investment_returns_recommendation(self):
        from src.mcp_server import server
        from src.schemas.models import ResearchResult

        fake = ResearchResult(question="q", steps=[], recommendation="HOLD")
        with patch.object(server, "run_research", new=AsyncMock(return_value=fake)):
            assert await server.research_investment("Risco de FIIs?") == "HOLD"

    def test_search_documents_returns_chunks(self):
        from src.mcp_server import server

        retriever = MagicMock()
        retriever.invoke.return_value = [MagicMock(page_content="chunk-a")]
        with patch.object(server, "get_retriever", return_value=retriever):
            assert server.search_documents("FIIs", k=1) == ["chunk-a"]

    def test_recall_memory_serializes_records(self):
        from src.mcp_server import server

        store = MagicMock()
        store.recall.return_value = [MemoryRecord(question="q", recommendation="HOLD")]
        with patch.object(server, "MemoryStore", return_value=store):
            out = server.recall_memory("FIIs", k=2)
        assert out[0]["recommendation"] == "HOLD"

    def test_remember_insight_persists(self):
        from src.mcp_server import server

        store = MagicMock()
        store.remember.return_value = "mem-1"
        with patch.object(server, "MemoryStore", return_value=store):
            assert server.remember_insight("q", "HOLD") == "mem-1"
