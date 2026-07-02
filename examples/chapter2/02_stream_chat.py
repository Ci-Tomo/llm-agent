"""2.1.2 Stream Generation

ノートブック: notebooks/chapter2/01_openai_introduction.ipynb
"""

import argparse

from llm_agent.config import get_default_model, get_openai_client


def stream_chat(client, model: str, max_turns: int = 10) -> None:
    """ストリーミング応答の対話型チャット。"""
    history: list[dict[str, str]] = []
    for _ in range(max_turns):
        user_input = input("ユーザ入力: ")
        if user_input == "exit":
            break
        print(f"ユーザ: {user_input}")
        history.append({"role": "user", "content": user_input})

        stream = client.chat.completions.create(
            model=model, messages=history, stream=True
        )
        print("AI: ", end="")
        ai_content = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if chunk.choices[0].finish_reason == "stop":
                break
            if delta:
                print(delta, end="", flush=True)
                ai_content += delta
        print()
        history.append({"role": "assistant", "content": ai_content})


def demo_stream(client, model: str) -> None:
    """固定メッセージでストリーミングのデモ。"""
    messages = [{"role": "user", "content": "LLMエージェントについて1文で説明してください。"}]
    stream = client.chat.completions.create(model=model, messages=messages, stream=True)
    print("AI: ", end="")
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if chunk.choices[0].finish_reason == "stop":
            break
        if delta:
            print(delta, end="", flush=True)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI ストリーム生成")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="対話型チャットモードで実行",
    )
    args = parser.parse_args()

    client = get_openai_client()
    model = get_default_model()

    if args.interactive:
        stream_chat(client, model)
    else:
        demo_stream(client, model)


if __name__ == "__main__":
    main()
