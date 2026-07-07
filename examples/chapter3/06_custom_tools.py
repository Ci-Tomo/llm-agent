"""3.2.3 ツールを自作する

ノートブック: notebooks/chapter3/02_tools.ipynb
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

from fortune_tools import GetDate, GetFortune, get_date, get_fortune
from llm_agent.config import create_chat_openai


def create_llm() -> ChatOpenAI:
    return create_chat_openai()


def invoke_with_tools(tools: list, question: str, *, run_tool: bool = True) -> None:
    """ツールを紐付けた LLM に質問し、必要ならツール呼び出し結果を表示する。"""
    model = create_llm()
    model_with_tools = model.bind_tools(tools)
    response = model_with_tools.invoke([HumanMessage(content=question)])

    print(f"ContentString: {response.content}")
    print(f"ToolCalls: {response.tool_calls}")

    if run_tool and response.tool_calls:
        tool_map = {tool.name: tool for tool in tools}
        for tool_call in response.tool_calls:
            tool = tool_map[tool_call["name"]]
            result = tool.invoke(tool_call)
            print(f"ToolResult ({tool_call['name']}): {result}")


def demo_get_fortune() -> None:
    """おみくじツール単体のデモ。"""
    print(f"get_fortune 出力例: {get_fortune('10月22日')}")
    invoke_with_tools([GetFortune()], "10月22日の運勢を教えてください。")


def demo_today_fortune_without_date_tool() -> None:
    """日付ツールなしで「今日の運勢」を質問する。"""
    invoke_with_tools([GetFortune()], "今日の運勢を教えてください。")


def demo_get_date() -> None:
    """日付取得ツール単体のデモ。"""
    print(f"get_date 出力例: {get_date('今日')}")
    invoke_with_tools([GetDate()], "今日の日付を教えてください。。")


def demo_combined_tools() -> None:
    """おみくじと日付取得の複合ツールデモ。"""
    invoke_with_tools(
        [GetFortune(), GetDate()],
        "今日の運勢を教えてください。。",
        run_tool=False,
    )


def demo_tool_class() -> None:
    """Tool クラスによる別パターンのツール定義。"""
    get_date_tool = Tool(
        name="Get_date",
        description=(
            "今日の日付を取得する。インプットは 'date'です。"
            "'date' は、日付を取得する対象の日で、'今日', '明日', '明後日' という3種類の文字列から指定します。"
            "今日の日付を知りたい際は'今日'を入力します"
        ),
        func=get_date,
    )
    get_fortune_tool = Tool(
        name="Get_fortune",
        description=(
            "特定の日付の運勢を占う。インプットは 'date_string'です。"
            "'date_string' は、占いを行う日付で、mm月dd日 という形式です。"
            "1月1日の占いを行う際は'1月1日'を入力します"
        ),
        func=get_fortune,
    )

    print(get_date_tool.invoke("今日"))
    print(get_fortune_tool.invoke("10月23日"))


def main() -> None:
    print("=== おみくじツール ===")
    demo_get_fortune()

    print("\n=== 日付ツールなしで今日の運勢 ===")
    demo_today_fortune_without_date_tool()

    print("\n=== 日付取得ツール ===")
    demo_get_date()

    print("\n=== 複合ツール ===")
    demo_combined_tools()

    print("\n=== Tool クラス（appendix） ===")
    demo_tool_class()


if __name__ == "__main__":
    main()
