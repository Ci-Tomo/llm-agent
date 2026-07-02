"""2.1.3 画像を入力する (Vision)

ノートブック: notebooks/chapter2/01_openai_introduction.ipynb
"""

from llm_agent.config import get_default_model, get_openai_client
from llm_agent.openai_utils import image_to_content
from llm_agent.paths import CHAPTER2_DIR


def describe_single_image(client, model: str) -> None:
    """1枚の画像を説明する。"""
    image_path = CHAPTER2_DIR / "sample_image1.png"
    prompt = "これは何の画像ですか?"
    contents = [{"type": "text", "text": prompt}, image_to_content(image_path)]

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[{"role": "user", "content": contents}],
    )
    print(response.choices[0].message.content)


def compare_two_images(client, model: str) -> None:
    """2枚の画像の違いを説明する。"""
    image_path1 = CHAPTER2_DIR / "sample_image1.png"
    image_path2 = CHAPTER2_DIR / "sample_image2.png"
    prompt = "2枚の画像の違いを教えてください。"
    contents = [
        {"type": "text", "text": prompt},
        image_to_content(image_path1),
        image_to_content(image_path2),
    ]

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[{"role": "user", "content": contents}],
    )
    print(response.choices[0].message.content)


def main() -> None:
    client = get_openai_client()
    model = get_default_model()

    print("=== 1枚の画像 ===")
    describe_single_image(client, model)

    print("\n=== 2枚の画像の比較 ===")
    compare_two_images(client, model)


if __name__ == "__main__":
    main()
