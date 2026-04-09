"""Integration tests — validate the full pipeline with mocked LLM calls."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_end_to_end_research_flow(self, sample_documents):
        """Simulate the full Researcher → Analyst → Advisor flow."""
        research_output = (
            "FIIs renderam 12.3% (IFIX) em 2024. Risco principal: vacância."
        )
        analysis_output = (
            "**Risk score: 5/10**\n"
            "Risco moderado. Vacância é o principal fator. "
            "Retorno histórico de 12.3% ao ano é atrativo."
        )
        recommendation_output = (
            "**Verdict: HOLD**\n"
            "Adequado para perfil moderado. "
            "Alocação sugerida: 10-20% do portfólio."
        )

        mock_researcher = MagicMock()
        mock_researcher.ainvoke = AsyncMock(return_value=research_output)

        mock_analyst = MagicMock()
        mock_analyst.ainvoke = AsyncMock(return_value=analysis_output)

        mock_advisor = MagicMock()
        mock_advisor.ainvoke = AsyncMock(return_value=recommendation_output)

        mock_retriever = MagicMock()

        with (
            patch("src.agents.orchestrator.get_retriever", return_value=mock_retriever),
            patch("src.agents.orchestrator.build_researcher_chain", return_value=mock_researcher),
            patch("src.agents.orchestrator.build_analyst_chain", return_value=mock_analyst),
            patch("src.agents.orchestrator.build_advisor_chain", return_value=mock_advisor),
        ):
            from src.agents.orchestrator import run_research

            result = await run_research("Quais são os riscos de investir em FIIs?")

        # Verify pipeline order
        assert result.steps[0].agent == "researcher"
        assert result.steps[1].agent == "analyst"
        assert result.steps[2].agent == "advisor"

        # Verify content flows through the pipeline
        assert "12.3%" in result.steps[0].output
        assert "Risk score" in result.steps[1].output
        assert "HOLD" in result.recommendation

    @pytest.mark.asyncio
    async def test_api_to_pipeline_integration(self, test_client):
        """Test that the API endpoint correctly invokes the pipeline."""
        from src.schemas.models import AgentStep, ResearchResult

        expected = ResearchResult(
            question="Como diversificar com ETFs?",
            steps=[
                AgentStep(agent="researcher", output="ETFs replicam índices"),
                AgentStep(agent="analyst", output="Risco baixo"),
                AgentStep(agent="advisor", output="BUY para longo prazo"),
            ],
            recommendation="BUY para longo prazo",
        )

        with patch("src.main.run_research", new_callable=AsyncMock, return_value=expected):
            response = test_client.post(
                "/research",
                json={"question": "Como diversificar com ETFs?"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["steps"][0]["output"] == "ETFs replicam índices"
        assert data["recommendation"] == "BUY para longo prazo"
