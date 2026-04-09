"""Orchestrator — coordinates the multi-agent pipeline: Researcher → Analyst → Advisor."""

from src.agents.analyst import build_analyst_chain
from src.agents.advisor import build_advisor_chain
from src.agents.researcher import build_researcher_chain
from src.rag.retriever import get_retriever
from src.schemas.models import AgentStep, ResearchResult


async def run_research(question: str) -> ResearchResult:
    """Execute the full multi-agent pipeline for an investment research question.

    Pipeline:
        1. Researcher — retrieves relevant documents via RAG and summarizes findings
        2. Analyst   — evaluates risk and return based on the research
        3. Advisor   — synthesizes everything into an actionable recommendation
    """
    retriever = get_retriever()

    # Step 1: Research
    researcher_chain = build_researcher_chain(retriever)
    research = await researcher_chain.ainvoke(question)

    # Step 2: Analysis
    analyst_chain = build_analyst_chain()
    analysis = await analyst_chain.ainvoke({"research": research, "question": question})

    # Step 3: Recommendation
    advisor_chain = build_advisor_chain()
    recommendation = await advisor_chain.ainvoke({
        "research": research,
        "analysis": analysis,
        "question": question,
    })

    return ResearchResult(
        question=question,
        steps=[
            AgentStep(agent="researcher", output=research),
            AgentStep(agent="analyst", output=analysis),
            AgentStep(agent="advisor", output=recommendation),
        ],
        recommendation=recommendation,
    )
