"""Gradio の応用: 翻訳アプリケーション

ノートブック: notebooks/chapter2/03_gradio_introduction.ipynb
"""

import gradio as gr
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI

from llm_agent.config import get_default_model, load_env

TRANSLATION_PROMPT = """\
以下の文章を {language} に翻訳し、翻訳結果のみを返してください。
{source_text}
"""

LANGUAGES = ["日本語", "英語", "中国語", "ラテン語", "ギリシャ語"]


def create_runnable():
    load_env()
    llm = ChatOpenAI(model=get_default_model())
    prompt = PromptTemplate.from_template(TRANSLATION_PROMPT)
    return prompt | llm


def translate(source_text: str, language: str) -> str:
    """Runnable による翻訳。"""
    runnable = create_runnable()
    response = runnable.invoke(dict(source_text=source_text, language=language))
    return response.content


def build_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        source_text = gr.Textbox(label="翻訳元の文章")
        language = gr.Dropdown(label="言語", choices=LANGUAGES)
        button = gr.Button("翻訳")
        translated_text = gr.Textbox(label="翻訳結果")

        button.click(inputs=[source_text, language], outputs=translated_text, fn=translate)
    return demo


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()
