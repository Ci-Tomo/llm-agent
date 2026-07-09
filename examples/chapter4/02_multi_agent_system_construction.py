"""4.2 マルチエージェントシステムの構築

ノートブック: notebooks/chapter4/02_multi_agent_system_construction.ipynb

実行例（プロジェクトルートで）:
  source .venv/bin/activate
  pip install -r requirements.txt
  python examples/chapter4/02_multi_agent_system_construction.py chatbot
  python examples/chapter4/02_multi_agent_system_construction.py persona
  python examples/chapter4/02_multi_agent_system_construction.py sequential
  python examples/chapter4/02_multi_agent_system_construction.py parallel
  python examples/chapter4/02_multi_agent_system_construction.py supervisor
  python examples/chapter4/02_multi_agent_system_construction.py tools
  python examples/chapter4/02_multi_agent_system_construction.py
"""

from __future__ import annotations

import argparse
import functools
import json
from typing import Annotated, Literal

try:
    from langchain.prompts import SystemMessagePromptTemplate
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from pydantic import BaseModel, Field
    from typing_extensions import TypedDict
except ModuleNotFoundError as exc:
    raise SystemExit(
        "必要なパッケージが見つかりません。\n"
        "プロジェクトルートで次を実行してください:\n"
        "  source .venv/bin/activate\n"
        "  pip install -r requirements.txt\n"
        "  python examples/chapter4/02_multi_agent_system_construction.py"
    ) from exc

from llm_agent.config import create_chat_openai
from llm_agent.secrets import get_tavily_api_key

KENTA_TRAITS = """\
- アクティブで冒険好き
- 新しい経験を求める
- アウトドア活動を好む
- SNSでの共有を楽しむ
- エネルギッシュで社交的"""

MARI_TRAITS = """\
- 穏やかでリラックス志向
- 家族を大切にする
- 静かな趣味を楽しむ
- 心身の休養を重視
- 丁寧な生活を好む"""

YUTA_TRAITS = """\
- バランス重視
- 柔軟性がある
- 自己啓発に熱心
- 伝統と現代の融合を好む
- 多様な経験を求める"""

MEMBER_TRAITS = {
    "kenta": KENTA_TRAITS,
    "mari": MARI_TRAITS,
    "yuta": YUTA_TRAITS,
}


class ChatState(TypedDict):
    count: int
    messages: Annotated[list, add_messages]


class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]


class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str


class RouteSchema(BaseModel):
    next: Literal["kenta", "mari", "yuta"] = Field(..., description="次に発言する人")


def create_llm(*, temperature: float | None = None):
    """章のノートブックに合わせ gpt-4o を使う。"""
    kwargs: dict = {"model": "gpt-4o"}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return create_chat_openai(**kwargs)


def stream_messages(graph, inputs: dict) -> None:
    """グラフを実行し、各ノードの最終メッセージを表示する。"""
    for event in graph.stream(inputs):
        for value in event.values():
            if "count" in value:
                print(f"### ターン{value['count']} ###")
            if "next" in value:
                print(f"次に発言する人: {value['next']}")
            if "messages" in value and value["messages"]:
                value["messages"][-1].pretty_print()


def build_basic_chatbot_graph():
    """4.2.2: 基本的なチャットボット。"""
    llm = create_llm(temperature=0)

    def chatbot(state: ChatState):
        messages = [llm.invoke(state["messages"])]
        return {"messages": messages, "count": state["count"] + 1}

    graph_builder = StateGraph(ChatState)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    return graph_builder.compile()


def build_persona_chatbot_graph():
    """4.2.2: ペルソナ付きチャットボット。"""
    llm = create_llm(temperature=0)

    def chatbot(state: ChatState):
        system_message = SystemMessage(
            "あなたは、元気なエンジニアです。元気に返答してください。"
        )
        messages = [llm.invoke([system_message, *state["messages"]])]
        return {"messages": messages, "count": state["count"] + 1}

    graph_builder = StateGraph(ChatState)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    return graph_builder.compile()


def agent_with_persona(state: MessagesState, name: str, traits: str):
    """ペルソナを持つエージェントノード。"""
    llm = create_llm()
    system_message_template = SystemMessagePromptTemplate.from_template(
        "あなたの名前は{name}です。\nあなたの性格は以下のとおりです。\n\n{traits}"
    )
    system_message = system_message_template.format(name=name, traits=traits)
    message = HumanMessage(
        content=llm.invoke([system_message, *state["messages"]]).content,
        name=name,
    )
    return {"messages": [message]}


def build_member_agents():
    """3人のエージェント関数を返す。"""
    kenta = functools.partial(agent_with_persona, name="kenta", traits=KENTA_TRAITS)
    mari = functools.partial(agent_with_persona, name="mari", traits=MARI_TRAITS)
    yuta = functools.partial(agent_with_persona, name="yuta", traits=YUTA_TRAITS)
    return kenta, mari, yuta


def build_sequential_graph():
    """4.2.3: 3つのエージェントが順番に回答するシステム。"""
    kenta, mari, yuta = build_member_agents()

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("kenta", kenta)
    graph_builder.add_node("mari", mari)
    graph_builder.add_node("yuta", yuta)
    graph_builder.add_edge(START, "kenta")
    graph_builder.add_edge("kenta", "mari")
    graph_builder.add_edge("mari", "yuta")
    graph_builder.add_edge("yuta", END)
    return graph_builder.compile()


def build_parallel_graph():
    """4.2.3: 3つのエージェントが一斉に回答するシステム。"""
    kenta, mari, yuta = build_member_agents()

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("kenta", kenta)
    graph_builder.add_node("mari", mari)
    graph_builder.add_node("yuta", yuta)
    graph_builder.add_edge(START, "kenta")
    graph_builder.add_edge(START, "mari")
    graph_builder.add_edge(START, "yuta")
    graph_builder.add_edge("kenta", END)
    graph_builder.add_edge("mari", END)
    graph_builder.add_edge("yuta", END)
    return graph_builder.compile()


def build_supervisor_graph():
    """4.2.3: 監督者が次の発言者を選ぶシステム。"""
    llm = create_llm()
    kenta, mari, yuta = build_member_agents()

    def supervisor(state: SupervisorState):
        system_message = SystemMessagePromptTemplate.from_template(
            "あなたは以下の作業者間の会話を管理する監督者です：{members}。"
            "各メンバーの性格は以下の通りです。"
            "{traits_description}"
            "与えられたユーザーリクエストに対して、次に発言する人を選択してください。"
        )
        members = ", ".join(MEMBER_TRAITS.keys())
        traits_description = "\n".join(
            f"**{name}**\n{traits}" for name, traits in MEMBER_TRAITS.items()
        )
        formatted = system_message.format(
            members=members,
            traits_description=traits_description,
        )
        llm_with_format = llm.with_structured_output(RouteSchema)
        next_speaker = llm_with_format.invoke([formatted, *state["messages"]]).next
        return {"next": next_speaker}

    graph_builder = StateGraph(SupervisorState)
    graph_builder.add_node("supervisor", supervisor)
    graph_builder.add_node("kenta", kenta)
    graph_builder.add_node("mari", mari)
    graph_builder.add_node("yuta", yuta)
    graph_builder.add_edge(START, "supervisor")
    graph_builder.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {"kenta": "kenta", "mari": "mari", "yuta": "yuta"},
    )
    for member in ("kenta", "mari", "yuta"):
        graph_builder.add_edge(member, END)
    return graph_builder.compile()


class ToolNode:
    """ツール呼び出しを実行するノード。"""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, state: MessagesState):
        messages = state.get("messages", [])
        if not messages:
            raise ValueError("入力にメッセージが見つかりません")

        message = messages[-1]
        tool_messages = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            tool_messages.append(
                ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": tool_messages}


def route_tools(state: MessagesState) -> Literal["tools", "__end__"]:
    """チャットボットの出力にツール呼び出しがあるか判定する。"""
    messages = state.get("messages", [])
    if not messages:
        raise ValueError(f"stateにツールに関するメッセージが見つかりませんでした: {state}")

    ai_message = messages[-1]
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"


def build_tool_chatbot_graph():
    """4.2.4: Tavily 検索ツールを使うチャットボット。"""
    tavily_tool = TavilySearchResults(max_results=2, tavily_api_key=get_tavily_api_key())
    llm = create_llm()
    llm_with_tool = llm.bind_tools([tavily_tool])
    tool_node = ToolNode([tavily_tool])

    def chatbot(state: MessagesState):
        return {"messages": [llm_with_tool.invoke(state["messages"])]}

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges("chatbot", route_tools, ["tools", "__end__"])
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    return graph_builder.compile()


def demo_chatbot() -> None:
    """4.2.2: 基本的なチャットボット。"""
    graph = build_basic_chatbot_graph()
    stream_messages(
        graph,
        {"messages": [HumanMessage("こんにちは")], "count": 0},
    )


def demo_persona() -> None:
    """4.2.2: ペルソナ付きチャットボット。"""
    graph = build_persona_chatbot_graph()
    stream_messages(
        graph,
        {"messages": [HumanMessage("上手くデバッグができません")], "count": 0},
    )


def demo_sequential() -> None:
    """4.2.3: 順番に回答するマルチエージェント。"""
    graph = build_sequential_graph()
    stream_messages(
        graph,
        {"messages": [HumanMessage("休日の過ごし方について、建設的に議論してください。")]},
    )


def demo_parallel() -> None:
    """4.2.3: 一斉に回答するマルチエージェント。"""
    graph = build_parallel_graph()
    stream_messages(
        graph,
        {"messages": [HumanMessage("休日の過ごし方について、建設的に議論してください。")]},
    )


def demo_supervisor() -> None:
    """4.2.3: 監督者が発言者を選ぶマルチエージェント。"""
    graph = build_supervisor_graph()
    stream_messages(
        graph,
        {"messages": [HumanMessage("休日のまったりした過ごし方を教えて")]},
    )


def demo_tools() -> None:
    """4.2.4: ツールを使うチャットボット。"""
    graph = build_tool_chatbot_graph()
    stream_messages(
        graph,
        {"messages": [HumanMessage("今日の東京の天気を教えて")]},
    )


DEMOS = {
    "chatbot": demo_chatbot,
    "persona": demo_persona,
    "sequential": demo_sequential,
    "parallel": demo_parallel,
    "supervisor": demo_supervisor,
    "tools": demo_tools,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="4.2 マルチエージェントシステムの構築")
    parser.add_argument(
        "demo",
        nargs="?",
        choices=[*DEMOS.keys(), "all"],
        default="all",
        help="実行するデモ",
    )
    args = parser.parse_args()

    if args.demo == "all":
        for name, demo in DEMOS.items():
            print(f"=== {name} ===")
            demo()
            print()
    else:
        DEMOS[args.demo]()


if __name__ == "__main__":
    main()
