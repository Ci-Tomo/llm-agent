"""2.1.4 音声を扱う

ノートブック: notebooks/chapter2/01_openai_introduction.ipynb
"""

from llm_agent.config import get_openai_client
from llm_agent.paths import CHAPTER2_DIR, OUTPUT_DIR


def transcribe_audio(client) -> None:
    """Whisper による文字起こし。"""
    audio_path = CHAPTER2_DIR / "sample_audio.mp3"
    with audio_path.open("rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", file=f, temperature=0.0
        )
    print("文字起こし:", transcription.text)


def transcribe_with_prompt(client) -> None:
    """プロンプト付き文字起こし。"""
    audio_path = CHAPTER2_DIR / "sample_audio.mp3"
    prompt = "下垣内"
    with audio_path.open("rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            prompt=prompt,
            response_format="text",
            temperature=0.0,
        )
    print("プロンプト付き文字起こし:", transcription)


def synthesize_speech(client) -> None:
    """TTS による音声合成。"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    audio_output_path = OUTPUT_DIR / "output.mp3"
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input="こんにちは。私は AI アシスタントです！",
    ) as response:
        response.stream_to_file(audio_output_path)
    print(f"音声ファイルを保存しました: {audio_output_path}")


def main() -> None:
    client = get_openai_client()

    print("=== 文字起こし ===")
    transcribe_audio(client)

    print("\n=== プロンプト付き文字起こし ===")
    transcribe_with_prompt(client)

    print("\n=== 音声合成 ===")
    synthesize_speech(client)


if __name__ == "__main__":
    main()
