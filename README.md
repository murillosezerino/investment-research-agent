# Investment Research Agent

[![CI](https://github.com/murillosezerino/investment-research-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/murillosezerino/investment-research-agent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Docker](https://img.shields.io/badge/container-Docker-2496ED)
![MCP](https://img.shields.io/badge/protocol-MCP-6E56CF)

> A sequential RAG pipeline for investment research (Researcher, Analyst, Advisor) with a **long-term memory layer** and an **MCP server**, orchestrated with LangChain and served via FastAPI.

Three role-specialized stages (Researcher, Analyst, Advisor) run in sequence over a vector store of research material, each passing its structured output to the next. This is a linear LCEL pipeline, not autonomous agents with dynamic routing or tool-calling. Each session is persisted to **long-term memory**, so the system recalls related past conclusions to enrich new answers instead of starting cold every time. The same capabilities are exposed both over HTTP and as **MCP tools**, so any MCP client (Claude Desktop, IDEs, other agents) can use them.

## What this project explores

- **Sequential role-based orchestration** (Researcher, Analyst, Advisor) with structured handoff
- **Vector retrieval** (RAG) with ChromaDB
- **Long-term memory** — semantic, persistent episodic recall across sessions
- **MCP server** — agent capabilities surfaced as standard Model Context Protocol tools
- **Agent handoff** with structured intermediate outputs
- **API delivery** with FastAPI
- **Test coverage** including unit and integration tests

## Stack

`Python` · `LangChain` · `ChromaDB` · `MCP` · `FastAPI` · `OpenAI API` · `pytest` · `Docker`

## Architecture

```
                         ┌──────────────────────────┐
                         │   long-term memory        │
                         │   (ChromaDB, persistent)  │
                         └─────────┬──────────┬──────┘
                          recall   │          │  persist
                                   ▼          ▲
user query ─▶ [Researcher] ─▶ [Analyst] ─▶ [Advisor] ─▶ recommendation
                 │ RAG over source docs           ▲
                 ▼                                 │ memory-aware
            ChromaDB (corpus)            prior conclusions injected as context

Two delivery surfaces over the same core:
   • FastAPI  →  /research, /ingest, /memory/recall, /memory/stats
   • MCP server →  research_investment, search_documents, recall_memory, remember_insight
```

Before the pipeline runs, the orchestrator **recalls** the most semantically similar past sessions and replays them as context for the Advisor. After it runs, the new conclusion is **persisted** so future related questions build on it. The source-document corpus and the memory live in separate ChromaDB collections.

## What's inside

```
investment-research-agent/
├── src/
│   ├── agents/
│   │   ├── researcher.py     # RAG retrieval + summary
│   │   ├── analyst.py        # risk/return analysis
│   │   ├── advisor.py        # memory-aware recommendation
│   │   └── orchestrator.py   # recall → pipeline → persist
│   ├── rag/                  # embeddings, ingestion, retriever
│   ├── memory/               # MemoryStore + MemoryRecord (long-term memory)
│   ├── mcp_server/           # MCP server exposing the agent as tools
│   ├── schemas/              # Pydantic models
│   ├── config.py
│   └── main.py               # FastAPI app
└── tests/                    # unit + integration
```

## How to run

### HTTP API

```bash
pip install ".[dev]"
docker compose up -d                          # (optional) containerized API
python -m src.rag.ingestion ./data/sample     # or POST /ingest to build the vector store
uvicorn src.main:app --reload
```

Endpoints:

| Method | Path             | Purpose                                      |
|--------|------------------|----------------------------------------------|
| GET    | `/health`        | Liveness                                     |
| POST   | `/ingest`        | Index documents into the corpus              |
| POST   | `/research`      | Run the sequential research pipeline         |
| POST   | `/memory/recall` | Recall related past sessions                 |
| GET    | `/memory/stats`  | Count of sessions in long-term memory        |

### MCP server

Run over stdio for desktop clients:

```bash
investment-mcp            # console script
# or: python -m src.mcp_server.server
```

Register it in Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "investment-research": {
      "command": "investment-mcp",
      "env": { "OPENAI_API_KEY": "sk-..." }
    }
  }
}
```

Exposed MCP tools: `research_investment`, `search_documents`, `recall_memory`, `remember_insight`.

## Configuration

Copy `.env.example` to `.env`. Memory-related settings:

| Variable             | Default            | Description                          |
|----------------------|--------------------|--------------------------------------|
| `ENABLE_MEMORY`      | `true`             | Toggle the long-term memory layer    |
| `MEMORY_PERSIST_DIR` | `./data/memory`    | Where memory is persisted            |
| `MEMORY_COLLECTION`  | `research_memory`  | ChromaDB collection for memory       |
| `MEMORY_RECALL_K`    | `3`                | How many past sessions to recall     |

## Notes

This is exploratory work on RAG, long-term memory and MCP patterns. The pipeline is sequential (LCEL), not an autonomous multi-agent system with dynamic routing or tool-calling. The recommendations produced are not investment advice — they reflect what the LLM synthesizes from indexed material, with all the limitations that implies (hallucination, context window, model bias).

## Author

Murillo Sezerino — Data Engineer & Analytics
[murillosezerino.com](https://murillosezerino.com) · [LinkedIn](https://linkedin.com/in/murillosezerino)
