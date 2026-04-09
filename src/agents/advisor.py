"""Advisor agent — synthesizes research and analysis into an actionable recommendation."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import settings

ADVISOR_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a certified financial advisor. Given the original research and the "
        "risk analysis below, produce a clear, actionable investment recommendation.\n\n"
        "Research summary:\n{research}\n\n"
        "Risk analysis:\n{analysis}\n\n"
        "Your recommendation MUST include:\n"
        "1. **Verdict** — BUY, HOLD, SELL, or AVOID with a one-line justification\n"
        "2. **Investor profile** — which investor profiles this is suitable for "
        "(conservative / moderate / aggressive)\n"
        "3. **Allocation suggestion** — recommended portfolio percentage range\n"
        "4. **Key considerations** — 2-3 bullet points the investor should monitor\n"
        "5. **Disclaimer** — remind that this is not personalized financial advice\n\n"
        "Be concise and practical. Respond in the same language as the research.",
    ),
    ("human", "{question}"),
])


def build_advisor_chain():
    """Build the advisor chain that generates the final recommendation."""
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.2,
        openai_api_key=settings.openai_api_key,
    )

    chain = ADVISOR_PROMPT | llm | StrOutputParser()
    return chain
