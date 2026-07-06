"""Gradio の基礎: Interface

ノートブック: notebooks/chapter2/03_gradio_introduction.ipynb
"""

import gradio as gr


def text2text(text: str) -> str:
    """入力テキストを << >> で囲んで返す。"""
    return "<<" + text + ">>"


def build_demo() -> gr.Interface:
    input_text = gr.Text(label="入力")
    output_text = gr.Text(label="出力")
    return gr.Interface(inputs=input_text, outputs=output_text, fn=text2text)


def main() -> None:
    demo = build_demo()
    demo.launch(debug=True)


if __name__ == "__main__":
    main()
