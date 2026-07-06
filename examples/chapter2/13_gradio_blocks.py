"""Gradio の基礎: Blocks

ノートブック: notebooks/chapter2/03_gradio_introduction.ipynb
"""

import gradio as gr


def text2text(text: str) -> str:
    """入力テキストを << >> で囲んで返す。"""
    return "<<" + text + ">>"


def text2text_rich(text: str) -> str:
    """入力テキストを装飾して返す。"""
    top = "^" * len(text)
    bottom = "v" * len(text)
    return f" {top}\n<{text}>\n {bottom}"


def build_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        input_text = gr.Text(label="入力")
        button1 = gr.Button(value="Normal")
        button2 = gr.Button(value="Rich")
        output_text = gr.Text(label="出力")

        button1.click(inputs=input_text, outputs=output_text, fn=text2text)
        button2.click(inputs=input_text, outputs=output_text, fn=text2text_rich)
    return demo


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()
