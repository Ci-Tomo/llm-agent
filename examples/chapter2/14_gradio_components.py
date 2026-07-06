"""Gradio の基礎: 重要なコンポーネント

ノートブック: notebooks/chapter2/03_gradio_introduction.ipynb
"""

import gradio as gr


def build_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Audio(label="音声", type="filepath")
        gr.Checkbox(label="チェックボックス")
        gr.File(label="ファイル", file_types=["image"])
        gr.Number(label="数値")
        gr.Markdown(label="Markdown", value="# タイトル\n## サブタイトル\n本文")
        gr.Slider(label="スライダー", minimum=-10, maximum=10, step=0.5, interactive=True)
        gr.Textbox(label="テキストボックス")
    return demo


def main() -> None:
    build_demo().launch(height=1200)


if __name__ == "__main__":
    main()
