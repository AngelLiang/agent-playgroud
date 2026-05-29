"""
对话记忆 MVP —— 带文件持久化的 ReAct Agent。

在 001_react.py 的 ReAct 循环基础上，增加记忆系统：
  - 对话开始前：自动加载 MEMORY.md 和近两日的记忆，注入 system prompt
  - 对话结束后：自动将本轮交互摘要追加到 memory/YYYY-MM-DD.md
  - memory_search 工具：关键词搜索所有记忆文件
  - remember 工具：显式记录一条信息到记忆

参考设计：
  - HiClaw 记忆系统（文件化记忆 + 写入/注入时机）
  - QwenPaw 记忆系统（auto_memory_search + summarize）

文件结构：
  memory/
  ├── MEMORY.md              # 长期记忆（经验、教训、重要信息）
  └── YYYY-MM-DD.md          # 每日对话日志
"""

from __future__ import annotations

import datetime
import json
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
    """注册一个可调用的 ReAct 工具。"""

    def decorator(func: Callable[[str], str]) -> Callable:
        _tools[name] = {"fn": func, "description": description}
        return func

    return decorator


# ── 记忆管理器 ──────────────────────────────────────────────────────────────

class MemoryManager:
    """文件化的对话记忆管理器。

    负责记忆的读取（注入）和写入（持久化）。
    """

    def __init__(self, base_dir: str = "memory") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._today = datetime.date.today()

    # ── 路径工具 ──────────────────────────────────────────────────────────

    @property
    def long_term_path(self) -> Path:
        return self.base_dir / "MEMORY.md"

    @property
    def today_path(self) -> Path:
        return self.base_dir / f"{self._today.isoformat()}.md"

    @property
    def yesterday_path(self) -> Path:
        yesterday = self._today - datetime.timedelta(days=1)
        return self.base_dir / f"{yesterday.isoformat()}.md"

    # ── 记忆注入（对话开始前）────────────────────────────────────────────────

    def load_context(self) -> str:
        """加载记忆上下文，用于注入 system prompt。

        返回：
          - MEMORY.md 全文（如果存在）
          - 今天 + 昨天的记忆文件摘要
        组合为一个记忆上下文字符串。
        """
        parts: list[str] = []

        # 1. 长期记忆（最重要，放在最前面）
        if self.long_term_path.exists():
            content = self.long_term_path.read_text(encoding="utf-8").strip()
            if content:
                parts.append(f"## 长期记忆（经验教训）\n\n{content}")

        # 2. 昨天的记忆
        if self.yesterday_path.exists():
            content = self.yesterday_path.read_text(encoding="utf-8").strip()
            if content:
                parts.append(f"## 昨天的对话记录\n\n{content}")

        # 3. 今天的记忆（当前会话之前的部分）
        if self.today_path.exists():
            content = self.today_path.read_text(encoding="utf-8").strip()
            if content:
                parts.append(f"## 今天的对话记录\n\n{content}")

        return "\n\n---\n\n".join(parts) if parts else ""

    # ── 记忆写入（对话结束后）────────────────────────────────────────────────

    def save_interaction(
        self,
        user_question: str,
        agent_answer: str,
        tool_calls: list[str] | None = None,
    ) -> None:
        """将一轮对话的关键信息追加写入今日记忆文件。

        写入格式为 Markdown 条目，包含时间戳、问题和回答摘要。
        """
        now = datetime.datetime.now().strftime("%H:%M")
        entry_parts = [f"- **[{now}]** 用户: {user_question}"]

        if tool_calls:
            entry_parts.append(f"  工具调用: {', '.join(tool_calls)}")

        # 截取回答的前 300 字符作为摘要
        summary = agent_answer[:300] + "..." if len(agent_answer) > 300 else agent_answer
        entry_parts.append(f"  回答: {summary}")

        entry = "\n".join(entry_parts) + "\n"

        with open(self.today_path, "a", encoding="utf-8") as f:
            f.write(entry)

    def remember(self, content: str) -> str:
        """将一条信息显式追加到今日记忆文件。

        Args:
            content: 要记录的信息内容（Agent 提炼后的文字）

        Returns:
            确认消息
        """
        now = datetime.datetime.now().strftime("%H:%M")
        entry = f"- **[{now}]** 📌 {content}\n"

        with open(self.today_path, "a", encoding="utf-8") as f:
            f.write(entry)

        return f"已记录: {content}"

    # ── 记忆搜索 ──────────────────────────────────────────────────────────

    def search(self, query: str, max_lines: int = 10) -> str:
        """按关键词搜索所有记忆文件，返回匹配的片段。

        支持多关键词搜索：用空格分隔的关键词，匹配任一即命中。
        也支持完整短语匹配（当不分割时）。

        Args:
            query: 搜索关键词（空格分隔的多关键词会做 OR 匹配）
            max_lines: 最多返回的行数

        Returns:
            匹配的文本片段，每条附带来源文件路径
        """
        # 收集所有可搜索的记忆文件
        memory_files: list[Path] = []
        if self.long_term_path.exists():
            memory_files.append(self.long_term_path)
        if self.yesterday_path.exists():
            memory_files.append(self.yesterday_path)
        if self.today_path.exists():
            memory_files.append(self.today_path)
        # 也搜其他日期的记忆文件
        for f in sorted(self.base_dir.glob("*.md")):
            if f not in memory_files and f.name != "MEMORY.md":
                memory_files.append(f)

        # 多关键词：按空格拆分，过滤空字符串
        keywords = [kw for kw in query.strip().split() if kw]

        results: list[str] = []
        line_count = 0

        for mem_file in memory_files:
            lines = mem_file.read_text(encoding="utf-8").splitlines()
            matching_context: list[str] = []

            for i, line in enumerate(lines):
                line_lower = line.lower()
                # 任一关键词命中即匹配（OR 逻辑）
                matched = any(
                    kw.lower() in line_lower for kw in keywords
                ) if keywords else (query.lower() in line_lower)

                if matched:
                    # 返回匹配行及其上下文（前后各 1 行）
                    start = max(0, i - 1)
                    end = min(len(lines), i + 2)
                    ctx_lines = lines[start:end]
                    if matching_context:
                        matching_context.append("  ...")
                    matching_context.extend(ctx_lines)
                    line_count += end - start
                    if line_count >= max_lines:
                        break
                if line_count >= max_lines:
                    break

            if matching_context:
                results.append(
                    f"【来源: {mem_file.relative_to(self.base_dir)}】\n"
                    + "\n".join(matching_context)
                )

            if line_count >= max_lines:
                break

        if not results:
            return f"未找到与「{query}」相关的记忆"

        return "\n\n".join(results)


# ── 全局记忆管理器实例 ──────────────────────────────────────────────────────

_memory = MemoryManager()


# ── 记忆工具 ────────────────────────────────────────────────────────────────


@tool(
    "memory_search",
    "搜索 Agent 的记忆文件，查找与关键词相关的历史对话记录和经验。"
    "输入：一个简短的关键词或短语。"
    "返回：匹配的记忆片段及其来源文件。"
    "用于：在需要回忆之前讨论过的内容、用户偏好、或历史经验时调用。",
)
def memory_search(query: str) -> str:
    return _memory.search(query)


@tool(
    "remember",
    "将一条重要信息显式记录到今日记忆文件中。"
    "输入：一条经过提炼的信息字符串（应包含关键上下文，"
    "以便将来检索）。"
    "用于：用户明确要求记住某事、发现了值得保留的经验教训、"
    "或做出了重要决策时调用。",
)
def remember(content: str) -> str:
    return _memory.remember(content)


# ── 内置工具 ────────────────────────────────────────────────────────────────

import ast
import math


@tool(
    "calculate",
    "计算数学表达式。支持 + - * / ** // % 以及 math.* 函数"
    "（sqrt, sin, cos, log, ceil, floor, pi, e 等）。",
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
        result = eval(code, safe_dict)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


@tool(
    "get_current_time",
    "获取当前的日期和时间。输入：任意字符串（会被忽略）。",
)
def get_current_time(_: str) -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── ReAct 引擎（带记忆）─────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
你是一个带记忆能力的 ReAct 智能体，通过思考和调用工具来回答问题。

{memory_context}

可用工具：
{tool_descriptions}

每一步都必须按以下格式输出：

Thought: <你对当前状态的推理，以及下一步计划>
Action: <工具名称>
Action Input: <传给工具的输入字符串>

当你获得足够的信息后，把 Action 替换为 Final Answer：

Thought: <你对已获信息的总结>
Final Answer: <你对用户问题的最终答案>

记忆使用规则：
- 对话开始时，系统已经自动注入了相关的历史记忆（见上方"记忆上下文"）
- 如果需要回忆更具体的信息，使用 memory_search 工具进行搜索
- 当用户明确要求记住某事，或发现了值得保留的经验教训时，使用 remember 工具
- 每次对话结束后，系统会自动保存本轮交互摘要

规则：
- 可以按顺序使用多个工具
- Observation（观察结果）来自工具输出 —— 用它们来支撑你的推理
- 绝不要编造 Observation —— 等工具返回结果
- 如果某个工具失败了，尝试换一个方法或工具
- 引用历史记忆中的信息时，说明"根据之前的记录..."或类似表述
"""

MAX_STEPS = 10


def _build_system_prompt() -> str:
    """构建带记忆上下文的 system prompt。"""
    memory_context = _memory.load_context()
    if not memory_context:
        memory_context = "（暂无历史记忆。随着对话进行，系统会自动记录。）"

    descriptions = "\n".join(
        f"  {name}: {info['description']}" for name, info in _tools.items()
    )
    return SYSTEM_PROMPT_TEMPLATE.format(
        memory_context=memory_context,
        tool_descriptions=descriptions,
    )


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
    """将完整对话记录保存到 conversation_log 目录。"""
    tmp_dir = Path(__file__).parent / "conversation_log"
    tmp_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = tmp_dir / f"memory_conversation_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            {"question": question, "messages": conversation_log},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"对话记录已保存至: {filename}")


def react(question: str, model: str | None = None) -> str | None:
    """运行带记忆的 ReAct 循环并返回最终答案。

    Args:
        question: 用户提出的问题。
        model: OpenAI 模型名称。

    Returns:
        最终答案字符串；如果循环耗尽未能得出答案则返回 None。
    """
    model = model or os.getenv("MODEL") or "gpt-4o-mini"
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    # ── 记忆注入（对话开始前）─────────────────────────────────────────────
    system_prompt = _build_system_prompt()

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    conversation_log: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    print(f"\n{'=' * 60}")
    print(f"问：{question}")
    print(f"{'=' * 60}\n")

    tool_calls_in_turn: list[str] = []

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

        conversation_log.append({"role": "assistant", "content": text})

        parsed = _parse_step(text)

        # ── 输出最终答案 → 保存记忆 → 结束 ──
        if parsed["final_answer"]:
            final = parsed["final_answer"]
            print(f"\n{'=' * 60}")
            print(f"答：{final}")
            print(f"{'=' * 60}")

            # ── 记忆写入（对话结束后）──────────────────────────────────────
            _memory.save_interaction(
                user_question=question,
                agent_answer=final,
                tool_calls=tool_calls_in_turn or None,
            )
            print(f"\n📝 对话摘要已保存至: {_memory.today_path}")

            _save_conversation(conversation_log, question)
            return final

        # ── 执行行动，把观察结果喂回上下文 ──
        if parsed["action"] and parsed["action_input"] is not None:
            tool_calls_in_turn.append(parsed["action"])
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

    DEMOS = [
        (
            "自我介绍 + 记住偏好",
            "我叫小明，我喜欢用中文交流，最喜欢用 Python 写后端代码。请记住这些关于我的信息。",
        ),
        (
            "回忆偏好",
            "你还记得我叫什么名字、喜欢什么编程语言吗？请先搜索记忆再回答。",
        ),
        (
            "记录经验",
            "我在部署 Flask 应用到服务器时，遇到了 502 错误，"
            "后来发现是因为 gunicorn 的 worker 数量设置太少导致的。"
            "请帮我记住这个经验教训。",
        ),
        (
            "多步回忆 + 推理",
            "根据你的记忆，我之前遇到过什么部署问题？问题原因是什么？"
            "如果我下次部署 FastAPI 应用，你会建议我注意什么？",
        ),
    ]

    # 显示记忆加载状态
    memory_ctx = _memory.load_context()
    if memory_ctx:
        print("🧠 已加载历史记忆：")
        print(f"  - 长期记忆: {'有' if _memory.long_term_path.exists() else '无'}")
        print(f"  - 今日记录: {'有' if _memory.today_path.exists() else '无'}")
        print(f"  - 昨日记录: {'有' if _memory.yesterday_path.exists() else '无'}")
        print()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("选择一个演示问题，观察带记忆的 ReAct 循环过程：\n")
        for i, (tag, q) in enumerate(DEMOS, 1):
            # 截断过长的描述
            desc = q[:60] + "..." if len(q) > 60 else q
            print(f"  {i}. {tag} — {desc}")
        print(f"  {len(DEMOS) + 1}. ✏️  自定义问题")
        print()

        try:
            choice = int(input("请输入编号 [1-5]：").strip())
            if 1 <= choice <= len(DEMOS):
                query = DEMOS[choice - 1][1]
            else:
                query = input("请输入你的问题: ").strip()
        except (ValueError, IndexError):
            query = input("请输入你的问题: ").strip()

    print(f"\n>>> 问题: {query}")
    react(query)
