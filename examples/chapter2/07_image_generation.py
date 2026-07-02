"""2.1.5 画像を生成する (DALL-E)

ノートブック: notebooks/chapter2/01_openai_introduction.ipynb
"""

import base64

import requests

from llm_agent.config import get_openai_client
from llm_agent.paths import OUTPUT_DIR

PROMPT = "メタリックな球体"


def generate_from_url(client) -> None:
    """URL から画像を取得して保存。"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    response = client.images.generate(
        model="dall-e-3", prompt=PROMPT, n=1, size="1024x1024"
    )
    image_url = response.data[0].url
    image = requests.get(image_url, timeout=60).content
    output_path = OUTPUT_DIR / "output1.png"
    output_path.write_bytes(image)
    print(f"画像を保存しました: {output_path}")


def generate_from_base64(client) -> None:
    """base64 形式で画像を取得して保存。"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    response = client.images.generate(
        model="dall-e-3",
        prompt=PROMPT,
        n=1,
        size="1024x1024",
        response_format="b64_json",
    )
    image_b64 = response.data[0].b64_json
    output_path = OUTPUT_DIR / "output2.png"
    output_path.write_bytes(base64.b64decode(image_b64))
    print(f"画像を保存しました: {output_path}")
    print("revised_prompt:", response.data[0].revised_prompt)


def main() -> None:
    client = get_openai_client()

    print("=== URL 形式で生成 ===")
    generate_from_url(client)

    print("\n=== base64 形式で生成 ===")
    generate_from_base64(client)


if __name__ == "__main__":
    main()
