import base64
from pathlib import Path
from typing import Any


def image_to_content(image_path: Path, detail: str = "low") -> dict[str, Any]:
    """画像ファイルを OpenAI Vision API 用の content 形式に変換する。"""
    with image_path.open("rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    suffix = image_path.suffix.lower().lstrip(".")
    media_type = "png" if suffix == "png" else suffix

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/{media_type};base64,{image_base64}",
            "detail": detail,
        },
    }
