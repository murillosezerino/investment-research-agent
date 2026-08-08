"""Orchestrator — coordinates the multi-agent pipeline: Researcher → Analyst → Advisor.

The pipeline is wrapped in a long-term memory layer: before running, it recalls
semantically related past sessions and replays them as context for the Advisor;
after running, it persists the new conclusion so future sessions can build on it.
"""

from src.agents.advisor import build_advisor_chain
from src.agents.analyst import build_analyst_chain
from src.agents.researcher import build_researcher_chain
from src.config import settings
from src.memory import MemoryRecord, MemoryStore, format_memories
from src.rag.retriever import get_retriever
from src.schemas.models import AgentStep, ResearchResult


async def run_research(question: str, memory: MemoryStore | None = None) -> ResearchResult:
    """Execute the full multi-agent pipeline for an investment research question.

    Pipeline:
        0. Memory   — recalls related past sessions to enrich context (long-term memory)
        1. Researcher — retrieves relevant documents via RAG and summarizes findings
        2. Analyst   — evaluates risk and return based on the research
        3. Advisor   — synthesizes everything into an actionable recommendation
        4. Memory   — persists the new conclusion for future sessions
    """
    if memory is None and settings.enable_memory:
        memory = MemoryStore()

    # Step 0: Recall related past research from long-term memory
    recalled: list[MemoryRecord] = memory.recall(question) if memory else []
    prior_context = format_memories(recalled)

    retriever = get_retriever()

    # Step 1: Research
    researcher_chain = build_researcher_chain(retriever)
    research = await researcher_chain.ainvoke(question)

    # Step 2: Analysis
    analyst_chain = build_analyst_chain()
    analysis = await analyst_chain.ainvoke({"research": research, "question": question})

    # Step 3: Recommendation (memory-aware)
    advisor_chain = build_advisor_chain()
    recommendation = await advisor_chain.ainvoke({
        "research": research,
        "analysis": analysis,
        "question": question,
        "prior_context": prior_context,
    })

    result = ResearchResult(
        question=question,
        steps=[
            AgentStep(agent="researcher", output=research),
            AgentStep(agent="analyst", output=analysis),
            AgentStep(agent="advisor", output=recommendation),
        ],
        recommendation=recommendation,
        recalled_memories=[r.question for r in recalled],
    )

    # Step 4: Persist this session for future recall
    if memory:
        memory.remember(MemoryRecord(question=question, recommendation=recommendation))

    return result
