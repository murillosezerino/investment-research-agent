from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.agents.orchestrator import run_research
from src.rag.ingestion import ingest_documents
from src.schemas.models import (
    HealthResponse,
    IngestRequest,
    IngestResult,
    QuestionRequest,
    ResearchResult,
)

VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Investment Research Agent",
    description="Multi-agent RAG system for investment research powered by LangChain",
    version=VERSION,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", version=VERSION)


@app.post("/ingest", response_model=IngestResult)
async def ingest(request: IngestRequest):
    """Ingest documents from a directory into the vector store."""
    try:
        vector_store = ingest_documents(request.source_dir)
        count = vector_store._collection.count()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}") from e

    return IngestResult(
        documents_indexed=count,
        collection=vector_store._collection.name,
    )


@app.post("/research", response_model=ResearchResult)
async def research(request: QuestionRequest):
    """Run the multi-agent research pipeline on an investment question."""
    try:
        result = await run_research(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research pipeline failed: {e}") from e

    return result
