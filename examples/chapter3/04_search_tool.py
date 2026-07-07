"""3.2.1 検索ツール

ノートブック: notebooks/chapter3/02_tools.ipynb

実行例（プロジェクトルートで）:
  source .venv/bin/activate
  python examples/chapter3/04_search_tool.py
"""

try:
    from langchain_community.agent_toolkits.load_tools import load_tools
except ModuleNotFoundError as exc:
    raise SystemExit(
        "必要なパッケージが見つかりません。\n"
        "プロジェクトルートで次を実行してください:\n"
        "  source .venv/bin/activate\n"
        "  pip install -r requirements.txt\n"
        "  python examples/chapter3/04_search_tool.py"
    ) from exc

from langchain_core.messages import HumanMessage

from llm_agent.config import create_chat_openai
from llm_agent.secrets import get_serpapi_api_key

QUESTION = (
    "株式会社Elithの住所を教えてください。"
    "最新の公式情報として公開されているものを教えてください。"
)


def demo_without_tools() -> None:
    """ツールなしで LLM に質問する。"""
    model = create_chat_openai()
    result = model.invoke([HumanMessage(content=QUESTION)])
    print(result.content)


def demo_with_search_tool() -> None:
    """SerpAPI 検索ツールを LLM に紐付けて質問する。"""
    model = create_chat_openai()
    tools = load_tools(
        ["serpapi"],
        llm=model,
        serpapi_api_key=get_serpapi_api_key(),
    )
    model_with_tools = model.bind_tools(tools)

    response = model_with_tools.invoke([HumanMessage(content=QUESTION)])
    print(f"ContentString: {response.content}")
    print(f"ToolCalls: {response.tool_calls}")
    print(response)
    

    if response.tool_calls:
        search_tool = tools[0]
        result = search_tool.invoke(response.tool_calls[0]["args"])
        print(f"SearchResult: {result}")


def main() -> None:
    print("=== ツールなし ===")
    demo_without_tools()

    print("\n=== 検索ツールあり ===")
    demo_with_search_tool()

if __name__ == "__main__":
    main()
