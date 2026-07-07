"""3.2.2 プログラム実行ツール

ノートブック: notebooks/chapter3/02_tools.ipynb
"""

from langchain_core.messages import HumanMessage
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_openai import ChatOpenAI

from llm_agent.config import create_chat_openai

QUESTION = (
    "以下をPythonで実行した場合の結果を教えてください。"
    "print(1873648+9285928+3759182+2398597)"
)
PYTHON_CODE = "print(1873648+9285928+3759182+2398597)"


def create_llm() -> ChatOpenAI:
    return create_chat_openai()


def demo_without_tools() -> None:
    """ツールなしで LLM に質問する。"""
    model = create_llm()
    result = model.invoke([HumanMessage(content=QUESTION)])
    print(result.content)


def demo_actual_execution() -> None:
    """Python を直接実行した結果。"""
    exec(PYTHON_CODE)


def demo_with_python_repl() -> None:
    """PythonREPLTool を LLM に紐付けて質問する。"""
    model = create_llm()
    tools = [PythonREPLTool()]
    model_with_tools = model.bind_tools(tools)

    response = model_with_tools.invoke([HumanMessage(content=QUESTION)])
    print(f"ContentString: {response.content}")
    print(f"ToolCalls: {response.tool_calls}")

    if response.tool_calls:
        python_repl_tool = PythonREPLTool()
        result = python_repl_tool.invoke(response.tool_calls[0]["args"])
        print(f"REPLResult: {result}")


def main() -> None:
    print("=== ツールなし ===")
    demo_without_tools()

    print("\n=== Python 直接実行 ===")
    demo_actual_execution()

    print("\n=== PythonREPLTool あり ===")
    demo_with_python_repl()


if __name__ == "__main__":
    main()
