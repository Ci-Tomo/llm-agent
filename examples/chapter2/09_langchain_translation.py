"""2.2.3 翻訳アプリ

ノートブック: notebooks/chapter2/02_langchain_introduction.ipynb
"""

from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI

from llm_agent.config import create_chat_openai

TRANSLATION_PROMPT = """\
以下の文章を {language} に翻訳し、翻訳結果のみを返してください。
{source_text}
"""


def create_llm() -> ChatOpenAI:
    """LangChain の ChatOpenAI インスタンスを生成する。"""
    return create_chat_openai()


def translate(language: str, source_text: str) -> None:
    """PromptTemplate と Runnable による翻訳。"""
    llm = create_llm()
    prompt = PromptTemplate.from_template(TRANSLATION_PROMPT)
    runnable = prompt | llm
    response = runnable.invoke(dict(language=language, source_text=source_text))
    response.pretty_print()


def main() -> None:
    language = "日本語"
    source_text = "cogito, ergo sum\n"
    translate(language, source_text)


if __name__ == "__main__":
    main()
