"""2.2.4 テーブル作成アプリ

ノートブック: notebooks/chapter2/02_langchain_introduction.ipynb
"""

import csv

from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai.chat_models import ChatOpenAI

from llm_agent.config import get_default_model, load_env
from llm_agent.paths import OUTPUT_DIR


class CSVSaveToolInput(BaseModel):
    filename: str = Field(description="ファイル名")
    csv_text: str = Field(description="CSVのテキスト")


@tool("csv-save-tool", args_schema=CSVSaveToolInput)
def csv_save(filename: str, csv_text: str) -> bool:
    """CSV テキストをファイルに保存する"""
    try:
        rows = list(csv.reader(csv_text.splitlines()))
    except Exception:
        return False

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / filename
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"CSV ファイルを保存しました: {output_path}")
    return True


TABLE_PROMPT = """\
{user_input}

結果は CSV ファイルに保存してください。ただし、ファイル名は上記の内容から適切に決定してください。
"""


def create_llm_with_tool() -> ChatOpenAI:
    """CSV 保存ツールを紐づけた LLM を生成する。"""
    load_env()
    llm = ChatOpenAI(model=get_default_model())
    return llm.bind_tools(tools=[csv_save], tool_choice="csv-save-tool")


def create_table(user_input: str) -> bool:
    """LLM にテーブル作成を依頼し、CSV として保存する。"""
    llm_with_tool = create_llm_with_tool()
    prompt = PromptTemplate.from_template(TABLE_PROMPT)
    get_tool_args = lambda x: x.tool_calls[0]
    runnable = prompt | llm_with_tool | get_tool_args | csv_save
    return runnable.invoke(dict(user_input=user_input))


def main() -> None:
    user_input = (
        "フィボナッチ数列の番号と値を10番目まで表にまとめて、"
        "CSV ファイルに保存してください。"
    )
    result = create_table(user_input)
    print(result)


if __name__ == "__main__":
    main()
