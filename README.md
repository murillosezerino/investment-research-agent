# Investment Research Agent

> Technical study: a multi-agent RAG system for investment research, orchestrated with LangChain and served via FastAPI.

A focused exercise in building a multi-agent retrieval-augmented generation pipeline. Three specialized agents (Researcher, Analyst, Advisor) work in sequence over a vector store of research material, with explicit handoff between roles.

## What this project explores

- **Multi-agent orchestration** with three role-specialized agents
- **Vector retrieval** with ChromaDB
- **Agent handoff** with structured intermediate outputs
- **API delivery** with FastAPI
- **Test coverage** including unit and integration tests

## Stack

`Python` · `LangChain` · `ChromaDB` · `FastAPI` · `OpenAI API` · `pytest` · `Docker`

## Architecture

```
user query
    ↓
[Researcher]  →  retrieves relevant material from ChromaDB
    ↓
[Analyst]     →  structures findings, identifies signals
    ↓
[Advisor]     →  composes recommendation with caveats
    ↓
FastAPI response
```

## What's inside

```
investment-research-agent/
├── agents/
│   ├── researcher.py
│   ├── analyst.py
│   └── advisor.py
├── store/                # ChromaDB setup
├── api/                  # FastAPI endpoint
└── tests/                # Unit + integration
```

## How to run

```bash
pip install -r requirements.txt
docker-compose up -d                    # ChromaDB
python scripts/index_documents.py       # Build vector store
uvicorn api.main:app --reload
```

## Notes

This is exploratory work on agentic patterns. The recommendations produced are not investment advice — they reflect what the LLM synthesizes from indexed material, with all the limitations that implies (hallucination, context window, model bias).

## Status

Study repository. Pipeline runs end-to-end with sample documents.

## Author

Murillo Sezerino — Analytics Engineer · Data Engineer
[murillosezerino.com](https://murillosezerino.com) · [LinkedIn](https://linkedin.com/in/murillosezerino)
