"""2.2.5 Plan-and-Solve

ノートブック: notebooks/chapter2/02_langchain_introduction.ipynb
"""

import argparse

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai.output_parsers.tools import PydanticToolsParser
from pydantic import BaseModel, Field

from llm_agent.config import get_default_model, load_env

PLAN_AND_SOLVE_PROMPT = """\
ユーザの質問が複雑な場合は、アクションプランを作成し、その後に1つずつ実行する Plan-and-Solve 形式をとります。
これが必要と判断した場合は、Plan ツールによってアクションプランを保存してください。
"""

ACTION_PROMPT = """\
問題をアクションプランに分解して解いています。
これまでのアクションの結果と、次に行うべきアクションを示すので、実際にアクションを実行してその結果を報告してください。
# 問題
{problem}
# アクションプラン
{action_items}
# これまでのアクションの結果
{action_results}
# 次のアクション
{next_action}
"""

DEMO_PROBLEM = """\
ある製造工場では、1時間に200個の部品が生産されます。工場は1日8時間稼働し、1週間に5日間営業しています。\
生産された部品のうち5%は品質不良で廃棄されます。この工場では1ヶ月（4週間）に品質不良で廃棄される部品の総数を求めなさい。"""


class ActionItem(BaseModel):
    action_name: str = Field(description="アクション名")
    action_description: str = Field(description="アクションの詳細")


class Plan(BaseModel):
    """アクションプランを格納する"""

    problem: str = Field(description="問題の説明")
    actions: list[ActionItem] = Field(description="実行すべきアクションリスト")


class ActionResult(BaseModel):
    """実行時の考えと結果を格納する"""

    thoughts: str = Field(description="検討内容")
    result: str = Field(description="結果")


def create_llm() -> ChatOpenAI:
    """LangChain の ChatOpenAI インスタンスを生成する。"""
    load_env()
    return ChatOpenAI(model=get_default_model())


def build_planning_runnable(llm: ChatOpenAI):
    """Plan-and-Solve 用の Runnable を構築する。"""
    llm_action = llm.bind_tools([ActionResult], tool_choice="ActionResult")
    action_parser = PydanticToolsParser(tools=[ActionResult], first_tool_only=True)
    plan_parser = PydanticToolsParser(tools=[Plan], first_tool_only=True)

    action_prompt = PromptTemplate.from_template(ACTION_PROMPT)
    action_runnable = action_prompt | llm_action | action_parser

    def action_loop(action_plan: Plan) -> AIMessage:
        problem = action_plan.problem
        actions = action_plan.actions

        action_items = "\n".join(["* " + action.action_name for action in actions])
        action_results_str = ""
        for i, action in enumerate(actions):
            print("=" * 20)
            print(f"[{i + 1}/{len(actions)}]以下のアクションを実行します。")
            print(action.action_name)

            next_action = f"* {action.action_name}  \n{action.action_description}"
            response = action_runnable.invoke(
                dict(
                    problem=problem,
                    action_items=action_items,
                    action_results=action_results_str,
                    next_action=next_action,
                )
            )
            action_results_str += f"* {action.action_name}  \n{response.result}\n"
            print("-" * 10 + "検討内容" + "-" * 10)
            print(response.thoughts)
            print("-" * 10 + "結果" + "-" * 10)
            print(response.result)

        return AIMessage(action_results_str)

    def route(ai_message: AIMessage):
        if ai_message.response_metadata["finish_reason"] == "tool_calls":
            return plan_parser | action_loop
        return ai_message

    system_prompt = SystemMessage(PLAN_AND_SOLVE_PROMPT)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_prompt, MessagesPlaceholder(variable_name="history")]
    )
    llm_plan = llm.bind_tools(tools=[Plan])
    return chat_prompt | llm_plan | route


def demo_plan_and_solve(planning_runnable, problem: str) -> None:
    """1問の Plan-and-Solve デモを実行する。"""
    history = [HumanMessage(problem)]
    HumanMessage(problem).pretty_print()
    ai_message = planning_runnable.invoke(dict(history=history))
    ai_message.pretty_print()


def interactive_chat(planning_runnable, max_turns: int = 10) -> None:
    """対話型 Plan-and-Solve チャット（exit で終了）。"""
    history: list[HumanMessage | AIMessage] = []
    for _ in range(max_turns):
        user_input = input("ユーザ入力: ")
        if user_input == "exit":
            break
        human_message = HumanMessage(user_input)
        human_message.pretty_print()
        history.append(HumanMessage(user_input))
        ai_message = planning_runnable.invoke(dict(history=history))
        ai_message.pretty_print()
        history.append(ai_message)


def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain Plan-and-Solve")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="対話型チャットモードで実行",
    )
    args = parser.parse_args()

    llm = create_llm()
    planning_runnable = build_planning_runnable(llm)

    if args.interactive:
        interactive_chat(planning_runnable)
    else:
        demo_plan_and_solve(planning_runnable, DEMO_PROBLEM)


if __name__ == "__main__":
    main()
