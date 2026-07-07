from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI

from llm_agent.secrets import get_openai_api_key, load_env

__all__ = [
    "create_chat_openai",
    "create_openai_embeddings",
    "get_default_model",
    "get_openai_client",
    "load_env",
]


def get_openai_client() -> OpenAI:
    """検証済みの API キーで OpenAI クライアントを生成する。"""
    return OpenAI(api_key=get_openai_api_key())


def get_default_model() -> str:
    """デフォルトのチャットモデル名を返す。"""
    load_env()
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def create_chat_openai(**kwargs: Any) -> ChatOpenAI:
    """API キーを明示的に渡して ChatOpenAI を生成する。"""
    model = kwargs.pop("model", get_default_model())
    return ChatOpenAI(model=model, api_key=get_openai_api_key(), **kwargs)


def create_openai_embeddings(**kwargs: Any) -> OpenAIEmbeddings:
    """API キーを明示的に渡して OpenAIEmbeddings を生成する。"""
    return OpenAIEmbeddings(api_key=get_openai_api_key(), **kwargs)
