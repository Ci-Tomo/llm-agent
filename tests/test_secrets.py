import pytest

from llm_agent.secrets import (
    SecretNotConfiguredError,
    get_openai_api_key,
    get_secret,
    load_env,
)


def test_existing_env_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-from-shell-env")
    assert get_openai_api_key() == "sk-test-from-shell-env"


def test_rejects_placeholder_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "your-openai-api-key-here")
    with pytest.raises(SecretNotConfiguredError) as exc_info:
        get_openai_api_key()
    assert exc_info.value.name == "OPENAI_API_KEY"
    assert "your-openai" not in str(exc_info.value)


def test_rejects_empty_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    with pytest.raises(SecretNotConfiguredError):
        get_openai_api_key()


def test_optional_secret_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    assert get_secret("SERPAPI_API_KEY", required=False) is None


def test_secret_error_repr_hides_value() -> None:
    error = SecretNotConfiguredError("OPENAI_API_KEY", "hint")
    assert repr(error) == "SecretNotConfiguredError(name='OPENAI_API_KEY')"


def test_load_env_is_idempotent() -> None:
    load_env()
    load_env()
