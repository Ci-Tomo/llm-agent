"""2.2.2 チャットアプリ

ノートブック: notebooks/chapter2/02_langchain_introduction.ipynb
"""

import argparse

from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI

from llm_agent.config import get_default_model, load_env


def create_llm() -> ChatOpenAI:
    """LangChain の ChatOpenAI インスタンスを生成する。"""
    load_env()
    return ChatOpenAI(model=get_default_model())


def demo_single_turn(llm: ChatOpenAI) -> None:
    """1ターンのチャットを実行する。"""
    history = [HumanMessage("こんにちは")]
    ai_message = llm.invoke(history)
    ai_message.pretty_print()


def interactive_chat(llm: ChatOpenAI, max_turns: int = 10) -> None:
    """対話型チャット（exit で終了）。"""
    history: list[HumanMessage] = []
    for _ in range(max_turns):
        user_input = input("ユーザ入力: ")
        if user_input == "exit":
            break
        human_message = HumanMessage(user_input)
        human_message.pretty_print()
        history.append(HumanMessage(user_input))
        ai_message = llm.invoke(history)
        ai_message.pretty_print()
        history.append(ai_message)


def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain チャットアプリ")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="対話型チャットモードで実行",
    )
    args = parser.parse_args()

    llm = create_llm()

    if args.interactive:
        interactive_chat(llm)
    else:
        demo_single_turn(llm)


if __name__ == "__main__":
    main()
