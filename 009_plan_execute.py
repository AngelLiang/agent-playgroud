"""
Plan and Execute（计划与执行）模式 —— 对比 ReAct 的另一种 Agent 架构。

核心流程：
  1. Plan（计划） — LLM 先将复杂问题分解为可执行的步骤列表
  2. Execute（执行） — 按顺序执行每个步骤，收集所有结果
  3. Summarize（总结） — LLM 综合所有步骤结果，输出最终答案
  4. Replan（可选） — 如果执行结果不足以回答问题，重新规划

与 ReAct 的关键区别：
  - ReAct:  思考→行动→观察→思考→行动→观察→... （交替进行）
  - Plan-Execute: 计划→[步骤1, 步骤2, ...]→总结 （计划与执行分离）

优势：
  - 复杂多步任务可以提前规划，避免迷路
  - 执行阶段不需要模型参与，效率更高、成本更低
  - 计划可审查、可修改，更可控

演示的知识点：
  1. 计划生成 — LLM 将问题分解为 JSON 格式的步骤计划
  2. 批量执行 — 遍历计划中的每个步骤并调用工具
  3. 结果汇总 — 将执行结果反馈给 LLM 生成最终答案
  4. 重新规划 — 当执行结果不充分时触发 replan
"""

from __future__ import annotations

import ast
import datetime
import json
import math
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
    """注册一个可调用的工具。"""

    def decorator(func: Callable[[str], str]) -> Callable:
        _tools[name] = {"fn": func, "description": description}
        return func

    return decorator


# ── 内置工具 ────────────────────────────────────────────────────────────────


@tool(
    "calculate",
    "计算数学表达式。支持 + - * / ** // % 以及 math.* 函数"
    "（sqrt, sin, cos, log, ceil, floor, pi, e 等）。示例：'sqrt(3**2 + 4**2)'",
)
def calculate(expr: str) -> str:
    ALLOWED_NODES = {
        ast.Expression, ast.Constant, ast.Name, ast.Load,
        ast.UnaryOp, ast.BinOp, ast.Add, ast.Sub, ast.Mult,
        ast.Div, ast.Pow, ast.FloorDiv, ast.Mod, ast.USub,
        ast.Call, ast.Attribute,
    }

    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        return f"语法错误: {e}"

    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            return f"不允许的语法结构: {type(node).__name__}"
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
        return str(eval(code, safe_dict))
    except Exception as e:
        return f"计算错误: {e}"


@tool(
    "get_current_time",
    "获取当前的日期和时间。输入：任意字符串（会被忽略）。",
)
def get_current_time(_: str) -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool(
    "search_facts",
    "按关键词查询预先存储的事实。输入：一个简短的关键词字符串。",
)
def search_facts(query: str) -> str:
    facts: dict[str, str] = {
        "法国首都": "巴黎",
        "法国人口": "约 6800 万（2024 年）",
        "法国时区": "欧洲中部时间（CET, UTC+1），夏令时为 CEST（UTC+2）",
        "python 作者": "Guido van Rossum",
        "python 诞生年份": "1991 年",
        "react 模式": (
            "ReAct = Reasoning + Acting，由 Yao 等人提出（ICLR 2023）。"
            "它将链式推理（Chain-of-Thought）与工具调用交替进行。"
        ),
        "plan execute 模式": (
            "Plan and Execute 是一种 Agent 架构，先将任务分解为计划，"
            "再逐步执行。代表论文：Plan-and-Solve (Wang et al., 2023)。"
        ),
        "生命的意义": "42（根据道格拉斯·亚当斯）",
        "中国首都": "北京",
        "中国人口": "约 14.1 亿（2024 年）",
        "美国首都": "华盛顿特区",
        "日本首都": "东京",
    }
    q = query.lower().replace(" ", "")
    for key, value in facts.items():
        key_normalized = key.replace(" ", "")
        if key_normalized in q or q in key_normalized:
            return value
    return f"没有找到关于「{query}」的事实"


# ── Plan and Execute 引擎 ───────────────────────────────────────────────────

MAX_PLAN_STEPS = 5

PLAN_SYSTEM_PROMPT = """\
你是一个 Plan and Execute 智能体。你的任务是将用户问题分解为可执行的步骤计划。

可用工具：
{tool_descriptions}

请按以下 JSON 格式输出执行计划：

```json
{{
  "plan": [
    {{
      "step": 1,
      "description": "步骤描述",
      "tool": "工具名称",
      "tool_input": "传给工具的输入"
    }}
  ]
}}
```

规则：
- 每个步骤必须使用一个可用工具，tool 字段必须是可用工具之一
- 步骤之间可以有依赖关系（后面的步骤可能需要前面的结果），但每个步骤的 tool_input 必须是自包含的
- plan 数组按执行顺序排列，最多 {max_steps} 个步骤
- 如果问题简单，可以只有 1-2 个步骤
- 如果问题超出工具能力范围，给出空 plan 并在最后的 summary 步骤中诚实说明

只输出 JSON，不要输出其他内容。"""


SUMMARIZE_SYSTEM_PROMPT = """\
你是一个 Plan and Execute 智能体。你已经执行了以下计划步骤并获得了结果。
请根据这些结果回答用户的原始问题。

规则：
- 如果所有步骤都成功，综合结果给出完整答案
- 如果部分步骤失败，基于已有信息尽力回答，并说明哪些信息缺失
- 如果结果不足以回答问题，诚实说明
- 回答要简洁、准确"""


def _build_plan_prompt() -> str:
    descriptions = "\n".join(
        f"  {name}: {info['description']}" for name, info in _tools.items()
    )
    return PLAN_SYSTEM_PROMPT.format(
        tool_descriptions=descriptions,
        max_steps=MAX_PLAN_STEPS,
    )


def _execute_tool(action: str, action_input: str) -> str:
    """执行工具并返回结果字符串。"""
    if action not in _tools:
        available = ", ".join(_tools.keys())
        return f"错误：未知工具「{action}」。可用工具：{available}"
    try:
        return _tools[action]["fn"](action_input)
    except Exception as e:
        return f"执行「{action}」时出错: {e}"


def _extract_json(text: str) -> dict | None:
    """从 LLM 输出中提取 JSON 对象。"""
    # 尝试直接解析
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试从 ```json ... ``` 代码块中提取
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试匹配第一个 { ... } 块
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _save_conversation(records: list[dict], question: str) -> None:
    """保存完整对话记录。"""
    log_dir = Path(__file__).parent / "conversation_log"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = log_dir / f"plan_execute_conversation_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({"question": question, "records": records}, f, ensure_ascii=False, indent=2)
    print(f"对话记录已保存至: {filename}")


def plan_and_execute(question: str, model: str | None = None) -> str | None:
    """运行 Plan and Execute 循环。

    1. Plan 阶段：LLM 生成 JSON 格式的执行计划
    2. Execute 阶段：按顺序执行计划中的每个步骤
    3. Summarize 阶段：LLM 综合所有结果生成最终答案
    """
    model = model or os.getenv("MODEL") or "gpt-4o-mini"
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    records: list[dict] = []

    print(f"\n{'=' * 60}")
    print(f"问：{question}")
    print(f"{'=' * 60}\n")

    # ── 阶段 1：Plan ──────────────────────────────────────────────────────
    print("📋 阶段 1：生成执行计划\n")

    plan_messages = [
        {"role": "system", "content": _build_plan_prompt()},
        {"role": "user", "content": f"请为以下问题生成执行计划：\n\n{question}"},
    ]

    plan_response = client.chat.completions.create(
        model=model,
        messages=plan_messages,
        temperature=0.1,
    )
    plan_text = plan_response.choices[0].message.content or ""
    records.append({"phase": "plan", "prompt": plan_messages, "output": plan_text})

    plan_data = _extract_json(plan_text)
    if not plan_data or "plan" not in plan_data:
        print("  ✗ 无法解析执行计划，模型输出：")
        print(f"  {plan_text[:300]}")
        _save_conversation(records, question)
        return None

    steps = plan_data["plan"]
    print(f"  计划共 {len(steps)} 个步骤：")
    for s in steps:
        print(f"    步骤 {s['step']}: [{s['tool']}] {s['description']}")
        print(f"           输入: {s['tool_input']}")
    print()

    # ── 阶段 2：Execute ────────────────────────────────────────────────────
    print("⚙️  阶段 2：执行计划\n")

    step_results: list[dict] = []
    for s in steps:
        step_num = s["step"]
        tool_name = s["tool"]
        tool_input = s["tool_input"]
        desc = s["description"]

        print(f"  ── 步骤 {step_num}: {desc} ──")
        print(f"     工具: {tool_name}({tool_input!r})")

        result = _execute_tool(tool_name, tool_input)
        print(f"     结果: {result}")

        step_results.append({
            "step": step_num,
            "description": desc,
            "tool": tool_name,
            "tool_input": tool_input,
            "result": result,
        })
    print()

    records.append({"phase": "execute", "steps": step_results})

    # ── 阶段 3：Summarize ──────────────────────────────────────────────────
    print("📝 阶段 3：总结答案\n")

    results_text = "\n".join(
        f"步骤 {r['step']}: {r['description']}\n"
        f"  工具: {r['tool']}({r['tool_input']!r})\n"
        f"  结果: {r['result']}"
        for r in step_results
    )

    summarize_messages = [
        {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"用户原始问题：{question}\n\n"
                f"执行计划及结果：\n{results_text}\n\n"
                f"请根据以上结果回答用户问题。"
            ),
        },
    ]

    summarize_response = client.chat.completions.create(
        model=model,
        messages=summarize_messages,
        temperature=0.3,
    )
    final_answer = summarize_response.choices[0].message.content or ""
    records.append({
        "phase": "summarize",
        "prompt": summarize_messages,
        "output": final_answer,
    })

    print(f"  {final_answer}")
    print(f"\n{'=' * 60}")
    print(f"答：{final_answer}")
    print(f"{'=' * 60}")

    _save_conversation(records, question)
    return final_answer


# ── CLI 入口 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    DEMOS = [
        ("🧮 简单计算", "计算 15 * 8 + 12 的结果"),
        ("📚 单步查询", "Python 的作者是谁？"),
        ("🔢 复合计算", "一个半径为 5 的圆，它的面积加上 100 是多少？"),
        ("🌍 多步查询", "中国首都和法国首都分别是什么？"),
        ("🔗 链式推理", "法国首都在哪个时区？先查首都，再查时区。"),
        ("📐 公式计算", "计算 sin(pi/6) + sqrt(25) 的值"),
    ]

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("Plan and Execute 模式演示\n")
        print("对比 ReAct（逐步思考+行动），Plan-Execute 先做全局计划再批量执行。\n")
        for i, (tag, q) in enumerate(DEMOS, 1):
            print(f"  {i}. {tag} — {q}")
        print(f"  {len(DEMOS) + 1}. ✏️  自定义问题")
        print()

        try:
            choice = int(input("请输入编号 [1-7]：").strip())
            if 1 <= choice <= len(DEMOS):
                query = DEMOS[choice - 1][1]
            else:
                query = input("请输入你的问题: ").strip()
        except (ValueError, IndexError):
            query = input("请输入你的问题: ").strip()

    print(f"\n>>> 问题: {query}")
    plan_and_execute(query)
