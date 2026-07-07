"""3.1.1 LLM に知識を与える

ノートブック: notebooks/chapter3/01_knowledge.ipynb
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from plant_documents import KUMATAROU_INFO
from llm_agent.config import create_chat_openai

QUESTION = "熊童子について教えてください。"

CONTEXT_PROMPT = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""


def create_llm() -> ChatOpenAI:
    return create_chat_openai()


def demo_without_context() -> None:
    """コンテキストなしで LLM に質問する。"""
    model = create_llm()
    result = model.invoke([HumanMessage(content=QUESTION)])
    print(result.content)


def demo_with_context() -> None:
    """プロンプトにコンテキストを埋め込んで質問する。"""
    prompt = ChatPromptTemplate.from_messages([("human", CONTEXT_PROMPT)])
    chain = prompt | create_llm()
    response = chain.invoke({"context": KUMATAROU_INFO, "question": QUESTION})
    print(response.content)


def main() -> None:
    print("=== コンテキストなし ===")
    demo_without_context()

    print("\n=== コンテキストあり ===")
    demo_with_context()


if __name__ == "__main__":
    main()
