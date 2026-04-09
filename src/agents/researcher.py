"""Researcher agent — retrieves relevant context from the vector store via RAG."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI

from src.config import settings

RESEARCHER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior investment researcher. Your job is to find and summarize "
        "the most relevant information from the provided documents to answer the "
        "user's question about investments.\n\n"
        "Retrieved context:\n{context}\n\n"
        "Rules:\n"
        "- Only use information present in the context\n"
        "- Cite specific data points, percentages and dates when available\n"
        "- If the context does not contain enough information, say so explicitly\n"
        "- Respond in the same language as the question",
    ),
    ("human", "{question}"),
])


def _format_docs(docs: list) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_researcher_chain(retriever: VectorStoreRetriever):
    """Build the RAG chain that retrieves documents and generates a research summary."""
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | RESEARCHER_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain
