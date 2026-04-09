from unittest.mock import AsyncMock, patch

from src.schemas.models import AgentStep, ResearchResult


class TestHealthEndpoint:
    def test_returns_ok(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestIngestEndpoint:
    def test_rejects_empty_dir(self, test_client, tmp_path):
        with patch("src.main.ingest_documents", side_effect=ValueError("No supported documents")):
            response = test_client.post("/ingest", json={"source_dir": str(tmp_path)})
        assert response.status_code == 400

    def test_ingest_success(self, test_client):
        mock_collection = type("Col", (), {"count": lambda self: 10, "name": "investments"})()
        mock_store = type("Store", (), {"_collection": mock_collection})()

        with patch("src.main.ingest_documents", return_value=mock_store):
            response = test_client.post("/ingest", json={"source_dir": "./data/sample"})

        assert response.status_code == 200
        data = response.json()
        assert data["documents_indexed"] == 10


class TestResearchEndpoint:
    def test_rejects_short_question(self, test_client):
        response = test_client.post("/research", json={"question": "FII?"})
        assert response.status_code == 422

    def test_research_success(self, test_client):
        mock_result = ResearchResult(
            question="Qual o risco dos fundos imobiliários?",
            steps=[
                AgentStep(agent="researcher", output="Pesquisa OK"),
                AgentStep(agent="analyst", output="Risco médio"),
                AgentStep(agent="advisor", output="HOLD"),
            ],
            recommendation="HOLD",
        )

        with patch("src.main.run_research", new_callable=AsyncMock, return_value=mock_result):
            response = test_client.post(
                "/research",
                json={"question": "Qual o risco dos fundos imobiliários?"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "Qual o risco dos fundos imobiliários?"
        assert len(data["steps"]) == 3
        assert data["recommendation"] == "HOLD"

    def test_handles_pipeline_error(self, test_client):
        with patch("src.main.run_research", new_callable=AsyncMock, side_effect=RuntimeError("LLM timeout")):
            response = test_client.post(
                "/research",
                json={"question": "Qual o risco dos fundos imobiliários?"},
            )
        assert response.status_code == 500
