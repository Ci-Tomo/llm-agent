from llm_agent.config import get_default_model, load_env
from llm_agent.paths import CHAPTER2_DIR, PROJECT_ROOT


def test_project_root_exists() -> None:
    assert PROJECT_ROOT.is_dir()


def test_chapter2_assets_exist() -> None:
    assert (CHAPTER2_DIR / "sample_image1.png").is_file()
    assert (CHAPTER2_DIR / "sample_image2.png").is_file()
    assert (CHAPTER2_DIR / "sample_audio.mp3").is_file()


def test_default_model() -> None:
    load_env()
    assert get_default_model() == "gpt-4o-mini"
