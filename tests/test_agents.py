from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.analyst import build_analyst_chain
from src.agents.advisor import build_advisor_chain
from src.agents.researcher import build_researcher_chain, _format_docs
from src.schemas.models import AgentStep, ResearchResult


class TestFormatDocs:
    def test_formats_multiple_docs(self, sample_documents):
        result = _format_docs(sample_documents)
        assert "---" in result
        assert "12.3%" in result
        assert "IPCA + 7.2%" in result

    def test_formats_empty_list(self):
        result = _format_docs([])
        assert result == ""


class TestResearcherChain:
    def test_chain_builds_successfully(self, mock_retriever):
        chain = build_researcher_chain(mock_retriever)
        assert chain is not None


class TestAnalystChain:
    def test_chain_builds_successfully(self):
        chain = build_analyst_chain()
        assert chain is not None


class TestAdvisorChain:
    def test_chain_builds_successfully(self):
        chain = build_advisor_chain()
        assert chain is not None


class TestResearchResult:
    def test_model_serialization(self):
        result = ResearchResult(
            question="Qual o risco de FIIs?",
            steps=[
                AgentStep(agent="researcher", output="Pesquisa concluída"),
                AgentStep(agent="analyst", output="Risco moderado"),
                AgentStep(agent="advisor", output="HOLD"),
            ],
            recommendation="HOLD — risco moderado com boa rentabilidade",
        )
        data = result.model_dump()
        assert data["question"] == "Qual o risco de FIIs?"
        assert len(data["steps"]) == 3
        assert data["steps"][0]["agent"] == "researcher"


class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_run_research_pipeline(self, mock_retriever):
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value="Mock output")

        with (
            patch("src.agents.orchestrator.get_retriever", return_value=mock_retriever),
            patch("src.agents.orchestrator.build_researcher_chain", return_value=mock_chain),
            patch("src.agents.orchestrator.build_analyst_chain", return_value=mock_chain),
            patch("src.agents.orchestrator.build_advisor_chain", return_value=mock_chain),
        ):
            from src.agents.orchestrator import run_research

            result = await run_research("Qual o risco de FIIs?")

        assert isinstance(result, ResearchResult)
        assert result.question == "Qual o risco de FIIs?"
        assert len(result.steps) == 3
        assert result.steps[0].agent == "researcher"
        assert result.steps[1].agent == "analyst"
        assert result.steps[2].agent == "advisor"
