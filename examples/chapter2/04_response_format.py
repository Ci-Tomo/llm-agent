"""2.1.2 response_format

ノートブック: notebooks/chapter2/01_openai_introduction.ipynb
"""

from pydantic import BaseModel, Field

from llm_agent.config import get_default_model, get_openai_client

SOURCE_TEXT = "吾輩は猫である。名前はまだない。"


class Translations(BaseModel):
    english: str = Field(description="英語の文章")
    french: str = Field(description="フランス語の文章")
    chinese: str = Field(description="中国語の文章")


def demo_json_object(client, model: str) -> None:
    """response_format=json_object による構造化出力。"""
    prompt = f"""\
以下に示す文章を英語・フランス語・中国語に翻訳してください。
ただし、アウトプットは後述するフォーマットの JSON 形式で出力してください。

# 文章
{SOURCE_TEXT}

# 出力フォーマット
以下に JSON Schema 形式のフォーマットを示します。このフォーマットに従うオブジェクトの形で出力してください。
{Translations.model_json_schema()}
"""
    response = client.chat.completions.create(
        temperature=0.0,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    translations = Translations.model_validate_json(response.choices[0].message.content)
    print("英語:", translations.english)
    print("フランス語:", translations.french)
    print("中国語:", translations.chinese)


def demo_parse(client, model: str) -> None:
    """beta.chat.completions.parse による構造化出力。"""
    prompt = f"""\
以下に示す文章を英語・フランス語・中国語に翻訳してください。
ただし、アウトプットは後述するフォーマットの JSON 形式で出力してください。

# 文章
{SOURCE_TEXT}

# 出力フォーマット
JSON Schema に従う形式で出力してください。
"""
    response = client.beta.chat.completions.parse(
        temperature=0.0,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=Translations,
    )
    translations = response.choices[0].message.parsed
    print("英語:", translations.english)
    print("フランス語:", translations.french)
    print("中国語:", translations.chinese)


def main() -> None:
    client = get_openai_client()
    model = get_default_model()

    print("=== response_format=json_object ===")
    demo_json_object(client, model)

    print("\n=== completions.parse ===")
    demo_parse(client, model)


if __name__ == "__main__":
    main()
