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


class IngestResult(BaseModel):
    documents_indexed: int
    collection: str


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
