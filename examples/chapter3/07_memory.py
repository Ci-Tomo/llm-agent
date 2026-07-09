"""3.4 記憶を持つエージェント

ノートブック: notebooks/chapter3/04_memory.ipynb

実行例（プロジェクトルートで）:
  source .venv/bin/activate
  python examples/chapter3/07_memory.py
"""

try:
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_community.agent_toolkits.load_tools import load_tools
    from langchain_community.chat_message_histories import ChatMessageHistory
except ModuleNotFoundError as exc:
    raise SystemExit(
        "必要なパッケージが見つかりません。\n"
        "プロジェクトルートで次を実行してください:\n"
        "  source .venv/bin/activate\n"
        "  pip install -r requirements.txt\n"
        "  python examples/chapter3/07_memory.py"
    ) from exc

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory

from llm_agent.config import create_chat_openai
from llm_agent.secrets import get_serpapi_api_key

ADDRESS_QUESTION = (
    "株式会社Elithの住所を教えてください。"
    "最新の公式情報として公開されているものを教えてください。"
)
FOLLOW_UP_QUESTION = "先ほど尋ねた会社は何の会社ですか？"

REACT_PROMPT = PromptTemplate(
    input_variables=["agent_scratchpad", "input", "tool_names", "tools", "chat_history"],
    template="""\
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Previous conversation history: {chat_history}
Question: {input}
Thought:{agent_scratchpad}""",
)

store: dict[str, ChatMessageHistory] = {}


def get_by_session_id(session_id: str) -> ChatMessageHistory:
    """セッション ID ごとに会話履歴を返す。"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def create_agent_with_history() -> RunnableWithMessageHistory:
    """ReAct エージェントとセッション別の会話履歴を構築する。"""
    model = create_chat_openai()
    tools = load_tools(
        ["serpapi"],
        llm=model,
        serpapi_api_key=get_serpapi_api_key(),
    )
    agent = create_react_agent(model, tools, REACT_PROMPT)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return RunnableWithMessageHistory(
        agent_executor,
        get_by_session_id,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


def invoke_with_session(
    agent_with_chat_history: RunnableWithMessageHistory,
    question: str,
    session_id: str,
) -> dict:
    """指定セッションでエージェントを実行する。"""
    return agent_with_chat_history.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}},
    )


def print_history(session_id: str) -> None:
    """セッションの会話履歴を表示する。"""
    print(f"--- session: {session_id} ---")
    print(get_by_session_id(session_id))


def demo_session1(agent_with_chat_history: RunnableWithMessageHistory) -> None:
    """セッション1: 住所を尋ね、続けて同じ会話を参照する。"""
    print("=== セッション1: 1回目の質問 ===")
    response = invoke_with_session(agent_with_chat_history, ADDRESS_QUESTION, "test-session1")
    print(f"Final Answer: {response['output']}")
    print_history("test-session1")

    print("\n=== セッション1: 2回目の質問（会話履歴あり） ===")
    response = invoke_with_session(agent_with_chat_history, FOLLOW_UP_QUESTION, "test-session1")
    print(f"Final Answer: {response['output']}")
    print_history("test-session1")


def demo_session2(agent_with_chat_history: RunnableWithMessageHistory) -> None:
    """セッション2: 履歴がない別セッションで同じ質問をする。"""
    print("=== セッション2: 履歴（空） ===")
    print_history("test-session2")

    print("\n=== セッション2: 同じ質問（会話履歴なし） ===")
    response = invoke_with_session(agent_with_chat_history, FOLLOW_UP_QUESTION, "test-session2")
    print(f"Final Answer: {response['output']}")
    print_history("test-session2")


def main() -> None:
    agent_with_chat_history = create_agent_with_history()
    demo_session1(agent_with_chat_history)
    print()
    demo_session2(agent_with_chat_history)


if __name__ == "__main__":
    main()
