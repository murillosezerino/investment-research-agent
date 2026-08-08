from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=10,
        examples=["Quais são os riscos de investir em fundos imobiliários em 2025?"],
    )


class IngestRequest(BaseModel):
    source_dir: str = Field(
        ...,
        examples=["./data/sample"],
    )


class AgentStep(BaseModel):
    agent: str
    output: str


class ResearchResult(BaseModel):
    question: str
    steps: list[AgentStep]
    recommendation: str
    recalled_memories: list[str] = Field(
        default_factory=list,
        description="Questions of past sessions recalled from long-term memory.",
    )


class IngestResult(BaseModel):
    documents_indexed: int
    collection: str


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


class MemoryRecallRequest(BaseModel):
    query: str = Field(..., min_length=3, examples=["riscos de fundos imobiliários"])
    k: int = Field(default=3, ge=1, le=20)


class MemoryRecallItem(BaseModel):
    question: str
    recommendation: str
    risk_score: str | None = None
    timestamp: str | None = None


class MemoryStats(BaseModel):
    stored: int
    collection: str
