import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

# Ensure test env doesn't hit real APIs
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./data/test_chroma")


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content=(
                "Fundos imobiliários (FIIs) são investimentos de renda variável que "
                "aplicam recursos em empreendimentos imobiliários. Em 2024, o IFIX "
                "apresentou rentabilidade de 12.3% ao ano. O risco principal é a "
                "vacância dos imóveis, que afeta diretamente os dividendos."
            ),
            metadata={"source": "fiis_guide.pdf", "page": 0},
        ),
        Document(
            page_content=(
                "Tesouro Direto IPCA+ é um título público indexado à inflação. "
                "Garante rentabilidade real acima do IPCA. Em abril de 2025, o "
                "Tesouro IPCA+ 2035 oferecia taxa de IPCA + 7.2% ao ano. "
                "Risco de marcação a mercado em resgates antecipados."
            ),
            metadata={"source": "renda_fixa.pdf", "page": 0},
        ),
        Document(
            page_content=(
                "ETFs (Exchange Traded Funds) replicam índices de mercado com baixo "
                "custo. O BOVA11 replica o Ibovespa com taxa de administração de "
                "0.10% ao ano. Adequado para diversificação passiva de longo prazo."
            ),
            metadata={"source": "etfs_guide.pdf", "page": 0},
        ),
    ]


@pytest.fixture
def mock_retriever(sample_documents):
    retriever = MagicMock()
    retriever.invoke.return_value = sample_documents
    retriever.ainvoke = AsyncMock(return_value=sample_documents)
    return retriever


@pytest.fixture
def mock_llm_response():
    return "Análise mock: Fundos imobiliários apresentam risco moderado com IFIX +12.3%."


@pytest.fixture
def tmp_docs_dir(tmp_path: Path):
    doc = tmp_path / "test_doc.txt"
    doc.write_text(
        "Fundos imobiliários renderam 12.3% em 2024. "
        "Tesouro IPCA+ oferece IPCA + 7.2% ao ano.",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def test_client():
    from src.main import app

    return TestClient(app)
