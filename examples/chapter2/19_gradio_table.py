"""Gradio の応用: テーブル作成アプリケーション

ノートブック: notebooks/chapter2/03_gradio_introduction.ipynb
"""

import csv

import gradio as gr
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai.chat_models import ChatOpenAI

from llm_agent.config import get_default_model, load_env

TABLE_PROMPT = """\
{user_input}
結果は CSV で作成し、csv2json-tool を利用して json に変換してください。
"""


class CSV2DFToolInput(BaseModel):
    csv_text: str = Field(description="CSVのテキスト")


@tool("csv2json-tool", args_schema=CSV2DFToolInput, return_direct=True)
def csv2json(csv_text: str) -> str:
    """CSV テキストを pandas DataFrame の JSON に変換する"""
    try:
        rows = list(csv.reader(csv_text.splitlines()))
        df = pd.DataFrame(rows[1:], columns=rows[0])
    except Exception:
        df = pd.DataFrame()
    return df.to_json()


def create_runnable():
    load_env()
    llm = ChatOpenAI(model=get_default_model())
    llm_with_tool = llm.bind_tools(tools=[csv2json], tool_choice="csv2json-tool")
    prompt = PromptTemplate.from_template(TABLE_PROMPT)
    get_tool_args = lambda x: x.tool_calls[0]
    return prompt | llm_with_tool | get_tool_args | csv2json


def create_df(user_input: str) -> pd.DataFrame:
    """LLM にテーブル作成を依頼し、DataFrame として返す。"""
    runnable = create_runnable()
    result = runnable.invoke(dict(user_input=user_input))
    json_str = result if isinstance(result, str) else result.content
    return pd.read_json(json_str)


def build_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        user_input = gr.Textbox(label="テーブルを作成したい内容のテキスト")
        button = gr.Button("実行")
        output_table = gr.DataFrame()

        button.click(inputs=user_input, outputs=output_table, fn=create_df)
    return demo


def main() -> None:
    build_demo().launch(height=1000)


if __name__ == "__main__":
    main()
