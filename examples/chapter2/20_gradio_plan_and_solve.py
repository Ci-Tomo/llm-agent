"""Gradio の応用: Plan-and-Solve チャットボット

ノートブック: notebooks/chapter2/03_gradio_introduction.ipynb
"""

import gradio as gr
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
    load_env()
    return ChatOpenAI(model=get_default_model())


def build_runnables(llm: ChatOpenAI):
    """Plan-and-Solve 用の Runnable を構築する。"""
    llm_action = llm.bind_tools([ActionResult], tool_choice="ActionResult")
    action_parser = PydanticToolsParser(tools=[ActionResult], first_tool_only=True)
    plan_parser = PydanticToolsParser(tools=[Plan], first_tool_only=True)

    action_prompt = PromptTemplate.from_template(ACTION_PROMPT)
    action_runnable = action_prompt | llm_action | action_parser

    def action_loop(action_plan: Plan):
        problem = action_plan.problem
        actions = action_plan.actions
        action_items = "\n".join(["* " + action.action_name for action in actions])
        action_results_str = ""
        for action in actions:
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
            yield response.thoughts, response.result

    system_prompt = SystemMessage(PLAN_AND_SOLVE_PROMPT)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_prompt, MessagesPlaceholder(variable_name="history")]
    )
    llm_plan = llm.bind_tools(tools=[Plan])
    planning_runnable = chat_prompt | llm_plan

    return planning_runnable, plan_parser, action_loop


def build_chat_fn(planning_runnable, plan_parser, action_loop):
    def chat(prompt: str, chat_history: list, langchain_history: list):
        chat_history = chat_history + [[prompt, None]]
        langchain_history.append(HumanMessage(content=prompt))
        response = planning_runnable.invoke(dict(history=langchain_history))

        if response.response_metadata["finish_reason"] != "tool_calls":
            chat_history[-1] = [prompt, response.content]
            langchain_history.append(AIMessage(content=response.content))
            yield "", chat_history, langchain_history
            return

        action_plan = plan_parser.invoke(response)
        action_items = "\n".join(
            ["* " + action.action_name for action in action_plan.actions]
        )
        assistant_text = f"**実行されるアクション**\n{action_items}"
        chat_history[-1] = [prompt, assistant_text]
        yield "", chat_history, langchain_history

        action_results_str = ""
        for i, (thoughts, result) in enumerate(action_loop(action_plan)):
            action_name = action_plan.actions[i].action_name
            action_results_str += f"* {action_name}  \n{result}\n"
            assistant_text += (
                f"\n\n### {action_name}\n"
                f"**思考過程:** {thoughts}\n"
                f"**結果:** {result}"
            )
            chat_history[-1] = [prompt, assistant_text]
            yield "", chat_history, langchain_history

        langchain_history.append(AIMessage(content=action_results_str))
        yield "", chat_history, langchain_history

    return chat


def build_demo() -> gr.Blocks:
    llm = create_llm()
    planning_runnable, plan_parser, action_loop = build_runnables(llm)
    chat = build_chat_fn(planning_runnable, plan_parser, action_loop)

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(label="Assistant", height=800)
        history = gr.State([])
        with gr.Row():
            with gr.Column(scale=9):
                user_input = gr.Textbox(lines=1, label="Chat Message")
            with gr.Column(scale=1):
                submit = gr.Button("Submit")
                clear = gr.ClearButton([user_input, chatbot, history])
        submit.click(
            chat,
            inputs=[user_input, chatbot, history],
            outputs=[user_input, chatbot, history],
        )
    return demo


def main() -> None:
    build_demo().launch(height=1000)


if __name__ == "__main__":
    main()
