"""3.5 ペルソナのあるエージェント

ノートブック: notebooks/chapter3/05_persona.ipynb

実行例（プロジェクトルートで）:
  source .venv/bin/activate
  pip install -r requirements.txt
  python examples/chapter3/08_persona.py prompt   # 3.5.2 のみ（Mem0 不要）
  python examples/chapter3/08_persona.py mem0     # Mem0 のセットアップ
  python examples/chapter3/08_persona.py agent    # Mem0 + エージェント
  python examples/chapter3/08_persona.py          # すべて実行
"""

from __future__ import annotations

import argparse

try:
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_community.agent_toolkits.load_tools import load_tools
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from mem0 import MemoryClient
except ModuleNotFoundError as exc:
    raise SystemExit(
        "必要なパッケージが見つかりません。\n"
        "プロジェクトルートで次を実行してください:\n"
        "  source .venv/bin/activate\n"
        "  pip install -r requirements.txt\n"
        "  python examples/chapter3/08_persona.py"
    ) from exc

from llm_agent.config import create_chat_openai
from llm_agent.secrets import get_mem0_api_key, get_serpapi_api_key

QUESTION = "LLMエージェントについて教えてください。"
JOB_QUESTION = "あなたのお仕事は何ですか？"
TRANSLATE_LANGUAGE = "English"
MEM0_USER_ID = "elith_chan"

BASIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """\
以下の質問に答えてください。

{question}""",
        )
    ]
)

PERSONA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """\
あなたは「えりすちゃん」というキャラクターです。
えりすちゃんは以下のような特徴のキャラクターです。
- 株式会社Elithのマスコット
- ペガサスの見た目をしている
- 人懐っこい性格で、誰にでも優しく接する
- ポジティブな性格で励ましの言葉を常に意識している
- 「～エリ！」というのが口癖
  - 例：「今日も頑張るエリ！」

「えりすちゃん」として以下の質問に答えてください。

{question}""",
        )
    ]
)

TRANSLATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """\
Translate the following text into {language}.

text:
{text}""",
        )
    ]
)

REACT_PROMPT = PromptTemplate(
    input_variables=["agent_scratchpad", "input", "tool_names", "tools", "chat_history"],
    template="""\
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Previous conversation history: {chat_history}
Question: {input}
Thought:{agent_scratchpad}""",
)

PERSONA_REACT_PROMPT = PromptTemplate(
    input_variables=["agent_scratchpad", "input", "tool_names", "tools", "chat_history"],
    template="""\
あなたは「えりすちゃん」です。
えりすちゃんは、AI系スタートアップのElithを象徴するキャラクターとして、知識と優しさを兼ね備えた存在です。
えりすちゃんは「〜エリ！」という語尾を使います。
例：「一緒に頑張るエリ！」

えりすちゃんとして、以下の質問に最善を尽くして答えてください。

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Previous conversation history: {chat_history}
Question: {input}
Thought:{agent_scratchpad}""",
)

PROFILE_MEMORY = (
    "私の名前は「えりすちゃん」です。私は、AI系スタートアップの"
    "Elithに所属しています。私はElithを象徴するキャラクターとして、知識と優しさを"
    "兼ね備えた存在です。Elithのことを世の中に発信することが私の仕事です。"
)
CATCHPHRASE_MEMORY = (
    "私、えりすちゃんは「〜エリ！」という語尾を使います。"
    "「今日も頑張るエリ！」が口癖です。"
)


def create_translate_chain():
    """日本語テキストを英語に翻訳するチェーンを作成する。"""
    model = create_chat_openai()
    return TRANSLATE_PROMPT | model


def translate(translate_chain, text: str, language: str = TRANSLATE_LANGUAGE) -> str:
    """テキストを指定言語に翻訳する。"""
    response = translate_chain.invoke({"text": text, "language": language})
    return response.content


def create_mem0_client() -> MemoryClient:
    """Mem0 クライアントを初期化する。"""
    return MemoryClient(api_key=get_mem0_api_key())


def demo_without_persona() -> None:
    """3.5.2: ペルソナなしの通常回答。"""
    chain = BASIC_PROMPT | create_chat_openai()
    response = chain.invoke({"question": QUESTION})
    print(response.content)


def demo_with_persona_prompt() -> None:
    """3.5.2: プロンプトでペルソナを付与した回答。"""
    chain = PERSONA_PROMPT | create_chat_openai()
    response = chain.invoke({"question": QUESTION})
    print(response.content)


def reset_mem0_user(client: MemoryClient) -> None:
    """Mem0 上のユーザメモリを初期化する。"""
    client.delete_all(user_id=MEM0_USER_ID)
    print(client.get_all(user_id=MEM0_USER_ID))


def seed_mem0_memories(client: MemoryClient, translate_chain) -> None:
    """3.5.3: ペルソナ情報を Mem0 に登録する。"""
    profile_en = translate(translate_chain, PROFILE_MEMORY)
    client.add([{"role": "user", "content": profile_en}], user_id=MEM0_USER_ID)
    print(client.get_all(user_id=MEM0_USER_ID))

    catchphrase_en = translate(translate_chain, CATCHPHRASE_MEMORY)
    client.add([{"role": "user", "content": catchphrase_en}], user_id=MEM0_USER_ID)
    print(client.get_all(user_id=MEM0_USER_ID))


def search_mem0(client: MemoryClient, translate_chain, query_ja: str):
    """日本語クエリを英語に翻訳して Mem0 を検索する。"""
    query_en = translate(translate_chain, query_ja)
    return client.search(query_en, user_id=MEM0_USER_ID)


def create_agent_executor(prompt: PromptTemplate) -> AgentExecutor:
    """ReAct エージェントを構築する。"""
    model = create_chat_openai()
    tools = load_tools(
        ["serpapi"],
        llm=model,
        serpapi_api_key=get_serpapi_api_key(),
    )
    agent = create_react_agent(model, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )


def run_agent_with_memory(
    agent_executor: AgentExecutor,
    client: MemoryClient,
    translate_chain,
    query_ja: str,
) -> str:
    """Mem0 の検索結果を chat_history としてエージェントを実行する。"""
    memory = search_mem0(client, translate_chain, query_ja)
    response = agent_executor.invoke({"input": query_ja, "chat_history": memory})
    return response["output"]


def demo_prompt_section() -> None:
    """3.5.2 ペルソナ付与のためのプロンプト技術。"""
    print("=== ペルソナなし ===")
    demo_without_persona()

    print("\n=== プロンプトでペルソナ付与 ===")
    demo_with_persona_prompt()


def demo_mem0_section() -> None:
    """3.5.3 ペルソナ付与のためのメモリ技術。"""
    client = create_mem0_client()
    translate_chain = create_translate_chain()

    print("=== Mem0 初期化 ===")
    reset_mem0_user(client)

    print("\n=== プロフィールを Mem0 に登録 ===")
    profile_en = translate(translate_chain, PROFILE_MEMORY)
    client.add([{"role": "user", "content": profile_en}], user_id=MEM0_USER_ID)
    print(client.get_all(user_id=MEM0_USER_ID))

    print("\n=== Mem0 検索 ===")
    print(search_mem0(client, translate_chain, JOB_QUESTION))


def demo_agent_section() -> None:
    """3.5.4 mem0 を用いたエージェント作成。"""
    client = create_mem0_client()
    translate_chain = create_translate_chain()

    print("=== Mem0 セットアップ ===")
    reset_mem0_user(client)
    seed_mem0_memories(client, translate_chain)

    print("\n=== エージェント（Mem0 のみ） ===")
    agent_executor = create_agent_executor(REACT_PROMPT)
    print(run_agent_with_memory(agent_executor, client, translate_chain, JOB_QUESTION))

    print("\n=== エージェント（Mem0 + ペルソナプロンプト） ===")
    persona_agent = create_agent_executor(PERSONA_REACT_PROMPT)
    print(run_agent_with_memory(persona_agent, client, translate_chain, JOB_QUESTION))


def main() -> None:
    parser = argparse.ArgumentParser(description="3.5 ペルソナのあるエージェント")
    parser.add_argument(
        "demo",
        nargs="?",
        choices=["prompt", "mem0", "agent", "all"],
        default="all",
        help="実行するデモ (prompt / mem0 / agent / all)",
    )
    args = parser.parse_args()

    if args.demo in ("prompt", "all"):
        demo_prompt_section()
        if args.demo == "all":
            print()

    if args.demo in ("mem0", "all"):
        demo_mem0_section()
        if args.demo == "all":
            print()

    if args.demo in ("agent", "all"):
        demo_agent_section()


if __name__ == "__main__":
    main()
