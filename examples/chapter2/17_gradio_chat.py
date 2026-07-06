"""Gradio の基礎: チャット UI

ノートブック: notebooks/chapter2/03_gradio_introduction.ipynb
"""

import argparse

import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI

from llm_agent.config import get_default_model, load_env


def create_llm() -> ChatOpenAI:
    load_env()
    return ChatOpenAI(model=get_default_model())


def history2messages(history: list[tuple[str, str]]) -> list[HumanMessage | AIMessage]:
    """Gradio の履歴を LangChain メッセージに変換する。"""
    messages: list[HumanMessage | AIMessage] = []
    for user, assistant in history:
        messages.append(HumanMessage(content=user))
        messages.append(AIMessage(content=assistant))
    return messages


def chat(message: str, history: list[tuple[str, str]]) -> str:
    """通常のチャット。"""
    llm = create_llm()
    messages = history2messages(history)
    messages.append(HumanMessage(content=message))
    response = llm.invoke(messages)
    return response.content


def stream_chat(message: str, history: list[tuple[str, str]]):
    """ストリーミングチャット。"""
    llm = create_llm()
    messages = history2messages(history)
    messages.append(HumanMessage(content=message))
    output = ""
    for chunk in llm.stream(messages):
        output += chunk.content
        yield output


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio チャット UI")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="ストリーミングチャットモードで起動",
    )
    args = parser.parse_args()

    fn = stream_chat if args.stream else chat
    demo = gr.ChatInterface(fn)
    demo.launch(debug=True)


if __name__ == "__main__":
    main()
