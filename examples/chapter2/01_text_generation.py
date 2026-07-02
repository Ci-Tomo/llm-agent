"""2.1.1 テキスト生成の基礎

ノートブック: notebooks/chapter2/01_openai_introduction.ipynb
"""

import argparse

from llm_agent.config import get_default_model, get_openai_client


def basic_generation(client, model: str) -> None:
    """単発のテキスト生成。"""
    response = client.chat.completions.create(
        temperature=0.0,
        model=model,
        messages=[{"role": "user", "content": "こんにちは"}],
    )
    print(response.choices[0].message.content)


def interactive_chat(client, model: str, max_turns: int = 10) -> None:
    """対話型チャット（exit で終了）。"""
    history: list[dict[str, str]] = []
    for _ in range(max_turns):
        user_input = input("ユーザ入力: ")
        if user_input == "exit":
            break
        print(f"ユーザ: {user_input}")
        history.append({"role": "user", "content": user_input})
        response = client.chat.completions.create(model=model, messages=history)
        content = response.choices[0].message.content
        print(f"AI: {content}")
        history.append({"role": "assistant", "content": content})


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI テキスト生成の基礎")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="対話型チャットモードで実行",
    )
    args = parser.parse_args()

    client = get_openai_client()
    model = get_default_model()

    if args.interactive:
        interactive_chat(client, model)
    else:
        basic_generation(client, model)


if __name__ == "__main__":
    main()
