# Investment Research Agent

Sistema multiagente RAG para pesquisa de investimentos com LangChain.

Três agentes de IA especializados colaboram em pipeline para responder perguntas sobre investimentos:

```
Pergunta → [Researcher] → [Analyst] → [Advisor] → Recomendação
                ↑
          Vector Store (RAG)
```

## Arquitetura

| Agente | Função | Output |
|---|---|---|
| **Researcher** | Recupera documentos relevantes via RAG e sintetiza achados | Resumo da pesquisa com dados citados |
| **Analyst** | Avalia risco e retorno com base na pesquisa | Risk score (1-10), fatores de risco, métricas |
| **Advisor** | Consolida tudo em recomendação acionável | Veredicto (BUY/HOLD/SELL), perfil de investidor, alocação |

### Stack

- **LangChain** — orquestração de agentes, chains LCEL, prompt templates
- **ChromaDB** — vector store para embeddings de documentos
- **OpenAI** — LLM (GPT-4o-mini) e embeddings (text-embedding-3-small)
- **FastAPI** — API REST servindo o pipeline multiagente
- **Pydantic v2** — validação de request/response
- **pytest** — testes unitários e de integração
- **Docker** — deploy containerizado
- **GitHub Actions** — pipeline CI/CD

## Quickstart

```bash
# Clone
git clone https://github.com/murillosezerino/investment-research-agent.git
cd investment-research-agent

# Setup
cp .env.example .env
# Edite o .env com sua chave da OpenAI

# Instalação
pip install ".[dev]"

# Execução
uvicorn src.main:app --reload

# Ingestão de documentos de exemplo
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_dir": "./data/sample"}'

# Testes
pytest -v
```

## Endpoints da API

### `GET /health`
Health check com versionamento.

### `POST /ingest`
Ingere documentos no vector store.

```json
{ "source_dir": "./data/sample" }
```

**Resposta:**

```json
{
  "documents_indexed": 15,
  "collection": "investments"
}
```

### `POST /research`
Executa o pipeline multiagente de pesquisa.

```json
{ "question": "Quais são os riscos de investir em fundos imobiliários em 2025?" }
```

**Resposta:**

```json
{
  "question": "Quais são os riscos de investir em fundos imobiliários em 2025?",
  "steps": [
    { "agent": "researcher", "output": "FIIs renderam 12.3% (IFIX) em 2024..." },
    { "agent": "analyst", "output": "Risk score: 5/10. Vacância é o principal fator..." },
    { "agent": "advisor", "output": "HOLD — adequado para perfil moderado, 10-20% do portfólio..." }
  ],
  "recommendation": "HOLD — risco moderado, adequado para perfil moderado..."
}
```

## Docker

```bash
docker compose up --build
```

## Estrutura do Projeto

```
├── src/
│   ├── main.py              # Aplicação FastAPI
│   ├── config.py             # Configurações via pydantic-settings
│   ├── agents/
│   │   ├── orchestrator.py   # Coordenação do pipeline multiagente
│   │   ├── researcher.py     # Agente de pesquisa com RAG
│   │   ├── analyst.py        # Agente de análise risco/retorno
│   │   └── advisor.py        # Agente de recomendação de investimento
│   ├── rag/
│   │   ├── ingestion.py      # Carregamento, splitting e indexação de documentos
│   │   ├── retriever.py      # Recuperação no vector store
│   │   └── embeddings.py     # Configuração de embeddings OpenAI
│   └── schemas/
│       └── models.py         # Modelos Pydantic
├── tests/
│   ├── conftest.py           # Fixtures compartilhadas
│   ├── test_rag.py           # Testes unitários do módulo RAG
│   ├── test_agents.py        # Testes unitários dos agentes
│   ├── test_api.py           # Testes dos endpoints da API
│   └── test_integration.py   # Testes de integração end-to-end
├── data/
│   └── sample/               # Documentos de exemplo (FIIs, renda fixa, ETFs)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Licença

MIT
