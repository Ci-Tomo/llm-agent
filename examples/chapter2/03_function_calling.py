"""2.1.2 Function Calling

ノートブック: notebooks/chapter2/01_openai_introduction.ipynb
"""

import json
import math

from pydantic import BaseModel, Field

from llm_agent.config import get_default_model, get_openai_client


class GCD(BaseModel):
    num1: int = Field(description="整数1")
    num2: int = Field(description="整数2")


class LCM(BaseModel):
    num1: int = Field(description="整数1")
    num2: int = Field(description="整数2")


def demo_basic_tool_call(client, model: str) -> None:
    """手動スキーマ定義による Function Calling。"""
    gcd_function = {
        "name": "gcd",
        "description": "最大公約数を求める",
        "parameters": {
            "type": "object",
            "properties": {
                "num1": {"type": "number", "description": "整数1"},
                "num2": {"type": "number", "description": "整数2"},
            },
            "required": ["num1", "num2"],
        },
    }
    tools = [{"type": "function", "function": gcd_function}]
    messages = [
        {"role": "user", "content": "50141 と 53599 の最大公約数を求めてください。"}
    ]

    response = client.chat.completions.create(
        model=model, messages=messages, tools=tools
    )
    choice = response.choices[0]
    print("content:", choice.message.content)
    print("finish_reason:", choice.finish_reason)
    print("tool_calls:", choice.message.tool_calls)

    function_info = choice.message.tool_calls[0].function
    args = json.loads(function_info.arguments)
    print("最大公約数:", math.gcd(args["num1"], args["num2"]))


def demo_pydantic_tool_call(client, model: str) -> None:
    """Pydantic スキーマによる Function Calling。"""
    gcd_function = {
        "name": "gcd",
        "description": "最大公約数を求める",
        "parameters": GCD.model_json_schema(),
    }
    tools = [{"type": "function", "function": gcd_function}]
    messages = [
        {"role": "user", "content": "50141 と 53599 の最大公約数を求めてください。"}
    ]

    response = client.chat.completions.create(
        model=model, messages=messages, tools=tools
    )
    parsed = GCD.model_validate_json(
        response.choices[0].message.tool_calls[0].function.arguments
    )
    print("Pydantic パース結果:", parsed)
    print("最大公約数:", math.gcd(parsed.num1, parsed.num2))


def demo_multiple_tools(client, model: str) -> None:
    """複数ツール（GCD / LCM）の利用。"""
    gcd_function = {
        "name": "gcd",
        "description": "最大公約数を求める",
        "parameters": GCD.model_json_schema(),
    }
    lcm_function = {
        "name": "lcm",
        "description": "最小公倍数を求める",
        "parameters": LCM.model_json_schema(),
    }
    tools = [
        {"type": "function", "function": gcd_function},
        {"type": "function", "function": lcm_function},
    ]
    messages = [
        {
            "role": "user",
            "content": "50141 と 53599 の最大公約数と最小公倍数を求めてください。",
        }
    ]

    response = client.chat.completions.create(
        model=model, messages=messages, tools=tools
    )
    choice = response.choices[0]
    if choice.finish_reason == "tool_calls":
        for tool in choice.message.tool_calls:
            if tool.function.name == "gcd":
                gcd_args = GCD.model_validate_json(tool.function.arguments)
                print(f"最大公約数: {math.gcd(gcd_args.num1, gcd_args.num2)}")
            elif tool.function.name == "lcm":
                lcm_args = LCM.model_validate_json(tool.function.arguments)
                print(f"最小公倍数: {math.lcm(lcm_args.num1, lcm_args.num2)}")
    elif choice.finish_reason == "stop":
        print("AI:", choice.message.content)


def main() -> None:
    client = get_openai_client()
    model = get_default_model()

    print("=== 基本の Function Calling ===")
    demo_basic_tool_call(client, model)

    print("\n=== Pydantic による Function Calling ===")
    demo_pydantic_tool_call(client, model)

    print("\n=== 複数ツールの利用 ===")
    demo_multiple_tools(client, model)


if __name__ == "__main__":
    main()
