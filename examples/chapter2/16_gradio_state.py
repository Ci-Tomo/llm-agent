"""Gradio の基礎: 状態を保持する

ノートブック: notebooks/chapter2/03_gradio_introduction.ipynb
"""

import gradio as gr


def build_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        username = gr.State("")
        text_input = gr.Text(label="ユーザ名")
        button1 = gr.Button("決定")
        button2 = gr.Button("自分の名前を表示")
        text_output = gr.Text(label="出力")
        button1.click(inputs=text_input, outputs=username, fn=lambda x: x)
        button2.click(inputs=username, outputs=text_output, fn=lambda x: x)
    return demo


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()
