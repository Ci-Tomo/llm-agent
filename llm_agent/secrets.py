"""API キー等の秘密情報を安全に取得するモジュール。"""

from __future__ import annotations

import os
import re
from typing import Final

from dotenv import load_dotenv

from llm_agent.paths import PROJECT_ROOT

_ENV_PATH: Final = PROJECT_ROOT / ".env"
_PLACEHOLDER_PATTERN = re.compile(
    r"(?:^your[-_]?|replace|changeme|example|dummy|test[-_]?key|<[^>]+>|xxx+)",
    re.IGNORECASE,
)


class SecretNotConfiguredError(RuntimeError):
    """必要な秘密情報が未設定、またはプレースホルダーのままの場合。"""

    def __init__(self, name: str, hint: str) -> None:
        self.name = name
        super().__init__(f"{name} が設定されていません。\n{hint}")

    def __repr__(self) -> str:
        return f"SecretNotConfiguredError(name={self.name!r})"


def load_env() -> None:
    """プロジェクトルートの .env を読み込む。

    既存の環境変数は上書きしない（本番・CI のシークレットを優先）。
    """
    if _ENV_PATH.is_file():
        load_dotenv(_ENV_PATH, override=False)


def _normalize_secret(value: str | None) -> str | None:
    if value is None:
        return None

    stripped = value.strip()
    if not stripped:
        return None
    if _PLACEHOLDER_PATTERN.search(stripped):
        return None
    return stripped


def get_secret(name: str, *, required: bool = True, setup_hint: str | None = None) -> str | None:
    """環境変数から秘密情報を取得する。値そのものはログや例外に含めない。"""
    load_env()
    value = _normalize_secret(os.getenv(name))
    if value is not None:
        return value

    if not required:
        return None

    hint = setup_hint or (
        f"  cp .env.example .env\n"
        f"  .env に {name} を設定するか、環境変数として渡してください。"
    )
    raise SecretNotConfiguredError(name, hint)


def get_openai_api_key() -> str:
    """OpenAI API キーを取得する。"""
    key = get_secret("OPENAI_API_KEY")
    assert key is not None
    return key


def get_serpapi_api_key() -> str:
    """SerpAPI キーを取得する。"""
    key = get_secret(
        "SERPAPI_API_KEY",
        setup_hint=(
            "  cp .env.example .env\n"
            "  .env に SERPAPI_API_KEY を設定してください（https://serpapi.com/）。"
        ),
    )
    assert key is not None
    return key


def get_mem0_api_key() -> str:
    """Mem0 API キーを取得する。"""
    key = get_secret(
        "MEM0_API_KEY",
        setup_hint=(
            "  cp .env.example .env\n"
            "  .env に MEM0_API_KEY を設定してください（https://mem0.ai/）。"
        ),
    )
    assert key is not None
    return key


def get_tavily_api_key() -> str:
    """Tavily API キーを取得する。"""
    key = get_secret(
        "TAVILY_API_KEY",
        setup_hint=(
            "  cp .env.example .env\n"
            "  .env に TAVILY_API_KEY を設定してください（https://tavily.com/）。"
        ),
    )
    assert key is not None
    return key
