"""MCP server exposing the investment-research agent over the Model Context Protocol.

This makes the agent's capabilities — document retrieval, the full sequential
pipeline, and the long-term memory layer — consumable as tools by any MCP client
(Claude Desktop, IDEs, other agents). The same engineering assets (semantic
retrieval + persistent memory) that power the HTTP API are surfaced here as a
standard, composable interface.

Run with stdio transport (the default for desktop clients):

    investment-mcp            # via the console script
    python -m src.mcp_server.server
"""

from mcp.server.fastmcp import FastMCP

from src.agents.orchestrator import run_research
from src.memory import MemoryRecord, MemoryStore
from src.rag.retriever import get_retriever

mcp = FastMCP("investment-research")


@mcp.tool()
async def research_investment(question: str) -> str:
    """Run the full Researcher → Analyst → Advisor pipeline and return a recommendation.

    Uses retrieval-augmented generation over the indexed research corpus and is
    enriched by related conclusions from long-term memory.
    """
    result = await run_research(question)
    return result.recommendation


@mcp.tool()
def search_documents(query: str, k: int = 4) -> list[str]:
    """Semantic search over the indexed research corpus. Returns the top-k chunks."""
    retriever = get_retriever(k=k)
    docs = retriever.invoke(query)
    return [doc.page_content for doc in docs]


@mcp.tool()
def recall_memory(query: str, k: int = 3) -> list[dict]:
    """Recall conclusions from past research sessions most related to ``query``."""
    store = MemoryStore()
    return [record.model_dump() for record in store.recall(query, k=k)]


@mcp.tool()
def remember_insight(question: str, recommendation: str, risk_score: str | None = None) -> str:
    """Persist an insight into long-term memory. Returns the stored memory id."""
    store = MemoryStore()
    return store.remember(
        MemoryRecord(question=question, recommendation=recommendation, risk_score=risk_score)
    )


def main() -> None:
    """Entry point — start the MCP server over stdio transport."""
    mcp.run()


if __name__ == "__main__":
    main()
