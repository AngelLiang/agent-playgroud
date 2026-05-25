"""
ReAct（Reasoning + Acting）模式 —— 最小化 Agent 实现。

核心循环：
  思考（Thought） → 行动（Action） → 观察（Observation） → （重复） → 最终答案（Final Answer）

演示的关键知识点：
  1. **Thought（思考）** — LLM 推理当前状态并规划下一步。
  2. **Action（行动）** — LLM 选择一个工具并传入参数。
  3. **Observation（观察）** — 工具执行结果作为上下文反馈给模型。
  4. **Final Answer（最终答案）** — LLM 获得足够信息后输出答案，循环终止。
  5. **工具注册表** — 基于装饰器的工具注册机制，引擎自动发现。
"""

from __future__ import annotations

import ast
import datetime
import json
import math
import operator
import os
import re
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── 工具注册表 ──────────────────────────────────────────────────────────────

_tools: dict[str, dict] = {}


def tool(name: str, description: str) -> Callable:
    """注册一个可调用的 ReAct 工具。

    用法::

        @tool("calculator", "计算数学表达式")
        def calculator(expr: str) -> str:
            ...
    """

    def decorator(func: Callable[[str], str]) -> Callable:
        _tools[name] = {"fn": func, "description": description}
        return func

    return decorator


# ── 内置工具 ────────────────────────────────────────────────────────────────


@tool(
    "calculate",
    "计算数学表达式。"
    "输入：数学表达式字符串。支持 + - * / ** // % 以及 math.* 函数"
    "（sqrt, sin, cos, log, ceil, floor, pi, e 等）。"
    "示例：'sqrt(3**2 + 4**2)'",
)
def calculate(expr: str) -> str:
    """通过 AST 遍历安全地计算数学表达式。"""
    # 允许的 AST 节点类型 — 其余全部拒绝。
    ALLOWED_NODES = {
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.UnaryOp,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.FloorDiv,
        ast.Mod,
        ast.USub,
        ast.Call,
        ast.Attribute,
    }

    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        return f"语法错误: {e}"

    # 拒绝危险结构。
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            return f"不允许的语法结构: {type(node).__name__}"
        # 只允许 math.* 属性访问，其他不允许。
        if isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name):
                return "不允许: 链式属性访问"
            if node.value.id != "math":
                return f"不允许: 对 '{node.value.id}' 进行属性访问"

    safe_dict = {
        "__builtins__": {},
        "math": math,
        **{k: v for k, v in vars(math).items() if not k.startswith("_")},
    }

    try:
        code = compile(tree, "<safe>", "eval")
        result = eval(code, safe_dict)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


@tool(
    "get_current_time",
    "获取当前的日期和时间。输入：任意字符串（会被忽略）。"
    "返回人类可读的时间戳。",
)
def get_current_time(_: str) -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool(
    "search_facts",
    "按关键词查询预先存储的事实。输入：一个简短的关键词字符串。"
    "知识库非常有限（硬编码）。",
)
def search_facts(query: str) -> str:
    """模拟知识查询 —— 生产环境中应替换为数据库或 API 调用。"""
    facts: dict[str, str] = {
        "法国首都": "巴黎",
        "python 作者": "Guido van Rossum",
        "react 模式": (
            "ReAct = Reasoning + Acting，由 Yao 等人提出（ICLR 2023）。"
            "它将链式推理（Chain-of-Thought）与工具调用交替进行。"
        ),
        "生命的意义": "42（根据道格拉斯·亚当斯）",
        "中国首都": "北京",
    }
    q = query.lower().strip()
    for key, value in facts.items():
        if key in q or q in key:
            return value
    return f"没有找到关于「{query}」的事实"


# ── ReAct 引擎 ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
你是一个 ReAct 智能体，通过思考和调用工具来回答问题。

可用工具：
{tool_descriptions}

每一步都必须按以下格式输出：

Thought: <你对当前状态的推理，以及下一步计划>
Action: <工具名称>
Action Input: <传给工具的输入字符串>

当你获得足够的信息后，把 Action 替换为 Final Answer：

Thought: <你对已获信息的总结>
Final Answer: <你对用户问题的最终答案>

规则：
- 可以按顺序使用多个工具。
- Observation（观察结果）来自工具输出 —— 用它们来支撑你的推理。
- 绝不要编造 Observation —— 等工具返回结果。
- 如果某个工具失败了，尝试换一个方法或工具。
"""

MAX_STEPS = 10


def _build_system_prompt() -> str:
    descriptions = "\n".join(
        f"  {name}: {info['description']}" for name, info in _tools.items()
    )
    return SYSTEM_PROMPT_TEMPLATE.format(tool_descriptions=descriptions)


def _parse_step(text: str) -> dict:
    """从模型输出中解析出结构化的步骤字段。"""
    result: dict = {
        "thought": "",
        "action": None,
        "action_input": None,
        "final_answer": None,
    }

    def _get(key: str) -> str | None:
        m = re.search(rf"^{key}:\s*(.*?)$", text, re.MULTILINE | re.DOTALL)
        if m:
            if key in ("Action Input", "Final Answer"):
                return m.group(1).strip()
            return m.group(1).split("\n")[0].strip()
        return None

    result["thought"] = _get("Thought") or ""
    result["action"] = _get("Action")
    result["action_input"] = _get("Action Input")
    result["final_answer"] = _get("Final Answer")

    return result


def _execute_tool(action: str, action_input: str) -> str:
    """执行工具并返回观察结果字符串。"""
    if action not in _tools:
        available = ", ".join(_tools.keys())
        return f"错误：未知工具「{action}」。可用工具：{available}"
    try:
        return _tools[action]["fn"](action_input)
    except Exception as e:
        return f"执行「{action}」时出错: {e}"


def _save_conversation(conversation_log: list[dict], question: str) -> None:
    """将完整对话记录保存到 tmp 目录，方便检查请求消息。"""
    tmp_dir = Path(__file__).parent / "conversation_log"
    tmp_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = tmp_dir / f"react_conversation_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            {"question": question, "messages": conversation_log},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"对话记录已保存至: {filename}")


def react(question: str, model: str | None = None) -> str | None:
    """运行 ReAct 循环并返回最终答案。

    Args:
        question: 用户提出的问题。
        model: OpenAI 模型名称。默认读取环境变量 MODEL，
               兜底为 gpt-4o-mini。

    Returns:
        最终答案字符串；如果循环耗尽未能得出答案则返回 None。
    """
    model = model or os.getenv("MODEL") or "gpt-4o-mini"
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    messages: list[dict] = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": question},
    ]

    # 记录完整的对话历史（含模型回复），用于事后审查。
    conversation_log: list[dict] = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": question},
    ]

    print(f"\n{'=' * 60}")
    print(f"问：{question}")
    print(f"{'=' * 60}\n")

    for step_idx in range(1, MAX_STEPS + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
        text = response.choices[0].message.content or ""

        print(f"─── 第 {step_idx} 步 ───")
        print(text)
        print()

        # 记录模型回复。
        conversation_log.append({"role": "assistant", "content": text})

        parsed = _parse_step(text)

        # ── 输出最终答案 → 结束 ──
        if parsed["final_answer"]:
            print(f"\n{'=' * 60}")
            print(f"答：{parsed['final_answer']}")
            print(f"{'=' * 60}")
            _save_conversation(conversation_log, question)
            return parsed["final_answer"]

        # ── 执行行动，把观察结果喂回上下文 ──
        if parsed["action"] and parsed["action_input"] is not None:
            observation = _execute_tool(parsed["action"], parsed["action_input"])
            print(f"  => 观察结果：{observation}\n")

            messages.append({"role": "assistant", "content": text})
            messages.append(
                {"role": "user", "content": f"Observation: {observation}"}
            )
            conversation_log.append(
                {"role": "user", "content": f"Observation: {observation}"}
            )
        else:
            print("  警告：未找到有效的 Action/FinalAnswer —— 停止循环。")
            break

    print("已达到最大步数，未能得出最终答案。")
    _save_conversation(conversation_log, question)
    return None


# ── CLI 入口 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # 演示问题 —— 每个问题触发不同的 ReAct 行为，方便观察循环过程。
    DEMOS = [
        ("🧮 计算", "计算 sin(30°) 的值"),                       # 单步工具调用
        ("⏰ 时间", "现在几点了？"),                                # 单步 + 不需要工具
        ("📚 事实", "ReAct 模式是谁提出的？"),                       # 知识查询 + 推理
        ("🔢 复合计算", "一个半径为 5 的圆的面积是多少？"),             # 多步（公式推理 + 计算）
        ("🔍 多步推理", "法国首都在什么时区？先查首都，再看时区。"),       # 多步工具链
    ]

    if len(sys.argv) > 1:
        # 命令行传参：uv run react.py "你的问题"
        query = " ".join(sys.argv[1:])
    else:
        # 交互菜单：选择一个演示问题
        print("选择一个演示问题，观察 ReAct 循环过程：\n")
        for i, (tag, q) in enumerate(DEMOS, 1):
            print(f"  {i}. {tag} — {q}")
        print(f"  {len(DEMOS) + 1}. ✏️  自定义问题")
        print()

        try:
            choice = int(input("请输入编号 [1-6]：").strip())
            if 1 <= choice <= len(DEMOS):
                query = DEMOS[choice - 1][1]
            else:
                query = input("请输入你的问题: ").strip()
        except (ValueError, IndexError):
            query = input("请输入你的问题: ").strip()

    print(f"\n>>> 问题: {query}")
    react(query)
