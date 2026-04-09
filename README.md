# Investment Research Agent

Multi-agent RAG system for investment research powered by LangChain.

Three specialized AI agents collaborate in a pipeline to answer investment questions:

```
User Question → [Researcher] → [Analyst] → [Advisor] → Recommendation
                    ↑
              Vector Store (RAG)
```

## Architecture

| Agent | Role | Output |
|---|---|---|
| **Researcher** | Retrieves relevant documents via RAG and summarizes findings | Research summary with cited data points |
| **Analyst** | Evaluates risk and return based on the research | Risk score (1-10), risk factors, key metrics |
| **Advisor** | Synthesizes everything into an actionable recommendation | Verdict (BUY/HOLD/SELL), investor profile, allocation |

### Tech Stack

- **LangChain** — agent orchestration, chains, prompt templates
- **ChromaDB** — vector store for document embeddings
- **OpenAI** — LLM (GPT-4o-mini) and embeddings (text-embedding-3-small)
- **FastAPI** — REST API serving the multi-agent pipeline
- **Pydantic v2** — request/response validation
- **pytest** — unit and integration tests
- **Docker** — containerized deployment
- **GitHub Actions** — CI/CD pipeline

## Quickstart

```bash
# Clone
git clone https://github.com/murillosezerino/investment-research-agent.git
cd investment-research-agent

# Setup
cp .env.example .env
# Edit .env with your OpenAI API key

# Install
pip install ".[dev]"

# Ingest sample documents
# Place PDF/TXT files in data/sample/, then:
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_dir": "./data/sample"}'

# Run
uvicorn src.main:app --reload

# Test
pytest -v
```

## API Endpoints

### `GET /health`
Health check.

### `POST /ingest`
Ingest documents into the vector store.

```json
{ "source_dir": "./data/sample" }
```

### `POST /research`
Run the multi-agent research pipeline.

```json
{ "question": "Quais são os riscos de investir em fundos imobiliários em 2025?" }
```

**Response:**

```json
{
  "question": "Quais são os riscos de investir em fundos imobiliários em 2025?",
  "steps": [
    { "agent": "researcher", "output": "..." },
    { "agent": "analyst", "output": "..." },
    { "agent": "advisor", "output": "..." }
  ],
  "recommendation": "HOLD — risco moderado, adequado para perfil moderado..."
}
```

## Docker

```bash
docker compose up --build
```

## Project Structure

```
├── src/
│   ├── main.py              # FastAPI application
│   ├── config.py             # Settings via pydantic-settings
│   ├── agents/
│   │   ├── orchestrator.py   # Multi-agent pipeline coordination
│   │   ├── researcher.py     # RAG-powered research agent
│   │   ├── analyst.py        # Risk-return analysis agent
│   │   └── advisor.py        # Investment recommendation agent
│   ├── rag/
│   │   ├── ingestion.py      # Document loading, splitting, indexing
│   │   ├── retriever.py      # Vector store retrieval
│   │   └── embeddings.py     # OpenAI embeddings config
│   └── schemas/
│       └── models.py         # Pydantic models
├── tests/
│   ├── conftest.py           # Shared fixtures
│   ├── test_rag.py           # RAG module unit tests
│   ├── test_agents.py        # Agent chain unit tests
│   ├── test_api.py           # API endpoint tests
│   └── test_integration.py   # End-to-end pipeline tests
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## License

MIT
