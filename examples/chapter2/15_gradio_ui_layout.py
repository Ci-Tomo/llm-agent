"""Gradio の基礎: UI の工夫

ノートブック: notebooks/chapter2/03_gradio_introduction.ipynb
"""

import argparse
import time

import gradio as gr


def build_layout_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        with gr.Accordion(label="アコーディオン"):
            gr.Text(value="アコーディオンの中身")
        with gr.Row():
            gr.Text(value="左")
            gr.Text(value="右")

        with gr.Row():
            with gr.Column():
                gr.Text(value="(0, 0)")
                gr.Text(value="(1, 0)")
            with gr.Column():
                gr.Text(value="(0, 1)")
                gr.Text(value="(1, 1)")

        with gr.Tab(label="タブ1"):
            gr.Text(value="コンテンツ1")
        with gr.Tab(label="タブ2"):
            gr.Text(value="コンテンツ2")
    return demo


def build_render_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        slider = gr.Slider(label="個数", minimum=0, maximum=10, step=1)

        @gr.render(inputs=slider)
        def render_blocks(value: int) -> None:
            for i in range(value):
                gr.Text(value=f"Block {i}")

    return demo


def iterative_output():
    """ジェネレータで段階的に出力する。"""
    for i in range(10):
        time.sleep(0.5)
        yield str(i)


def build_iterative_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        button = gr.Button("実行")
        output = gr.Text(label="出力")
        button.click(outputs=output, fn=iterative_output)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio UI の工夫")
    parser.add_argument(
        "demo",
        choices=["layout", "render", "iterative"],
        default="layout",
        nargs="?",
        help="起動するデモ (layout / render / iterative)",
    )
    args = parser.parse_args()

    if args.demo == "layout":
        build_layout_demo().launch(height=800)
    elif args.demo == "render":
        build_render_demo().launch()
    else:
        build_iterative_demo().launch()


if __name__ == "__main__":
    main()
