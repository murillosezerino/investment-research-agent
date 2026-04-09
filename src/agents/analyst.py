"""Analyst agent — evaluates risk and return based on the researcher's findings."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import settings

ANALYST_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a quantitative risk analyst specializing in investment products. "
        "Given the research summary below, produce a structured risk-return analysis.\n\n"
        "Research summary:\n{research}\n\n"
        "Your analysis MUST include:\n"
        "1. **Risk factors** — list each identified risk with severity (low/medium/high)\n"
        "2. **Return potential** — expected return profile based on available data\n"
        "3. **Risk score** — a single score from 1 (very low risk) to 10 (very high risk)\n"
        "4. **Key metrics** — any relevant financial metrics found in the research\n\n"
        "Be objective and data-driven. Respond in the same language as the research.",
    ),
    ("human", "{question}"),
])


def build_analyst_chain():
    """Build the analyst chain that evaluates risk and return."""
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )

    chain = ANALYST_PROMPT | llm | StrOutputParser()
    return chain
