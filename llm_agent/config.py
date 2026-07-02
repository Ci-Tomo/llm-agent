import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from llm_agent.paths import PROJECT_ROOT

_ENV_PATH = PROJECT_ROOT / ".env"


def load_env() -> None:
    """プロジェクトルートの .env を読み込む。"""
    load_dotenv(_ENV_PATH)


def get_openai_client() -> OpenAI:
    """環境変数から OpenAI クライアントを生成する。"""
    load_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY が設定されていません。\n"
            "  cp .env.example .env\n"
            "  .env に API キーを設定してください。"
        )
    return OpenAI()


def get_default_model() -> str:
    """デフォルトのチャットモデル名を返す。"""
    load_env()
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
