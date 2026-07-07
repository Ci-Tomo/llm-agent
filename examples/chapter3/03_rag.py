"""3.1.3 文書検索機能を持つ LLM

ノートブック: notebooks/chapter3/01_knowledge.ipynb
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from plant_documents import create_vectorstore
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


def demo_rag() -> None:
    """Retriever と LLM を組み合わせた RAG チェーン。"""
    vectorstore = create_vectorstore()
    retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)

    prompt = ChatPromptTemplate.from_messages([("human", CONTEXT_PROMPT)])
    model = create_llm()
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | model

    result = rag_chain.invoke(QUESTION)
    print(result.content)


def main() -> None:
    demo_rag()


if __name__ == "__main__":
    main()
