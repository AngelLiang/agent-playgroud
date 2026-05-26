"""
工具结果截断 —— QwenPaw 三层压缩防线的 Layer 0（post_acting 阶段）

核心概念：
  - 差异化截断：最近 N 条工具输出用宽松阈值，更早的用严格阈值
  - 完整内容落盘：截断的数据写入文件，不丢失
  - 工具调用成对保留：tool_use 和 tool_result 一起处理

参考 QwenPaw 的 _prune_tool_result() 和 AgentProfileConfig 中的
tool_result_pruning_config 配置项。

本 demo 接入 OpenAI SDK 模拟真实 ReAct 对话：
  1. Agent 读取 data/ 目录下的文件来分析日志
  2. 工具输出累积后触发截断
  3. 展示截断前后的对比效果
"""

from __future__ import annotations

import ast
import datetime
import json
import os
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# 0. 配置
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).parent / "data"

MAX_STEPS = 8                     # ReAct 最大步数
PRUNING_RECENT_N = 2              # 最近 N 条工具输出用宽松截断（QwenPaw 默认 2）
PRUNING_OLD_MAX_BYTES = 3000      # 旧消息截断字节数
PRUNING_RECENT_MAX_BYTES = 20000  # 最近消息截断字节数

@dataclass
class PruningConfig:
    """工具结果截断配置。"""
    enabled: bool = True
    pruning_recent_n: int = PRUNING_RECENT_N
    pruning_old_msg_max_bytes: int = PRUNING_OLD_MAX_BYTES
    pruning_recent_msg_max_bytes: int = PRUNING_RECENT_MAX_BYTES
    offload_retention_days: int = 7
    exempt_file_extensions: list[str] = field(default_factory=list)
    exempt_tool_names: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 工具注册表 + 内置工具
# ═══════════════════════════════════════════════════════════════════════════════

_tools: dict[str, dict] = {}

def tool(name: str, description: str):
    def decorator(func):
        _tools[name] = {"fn": func, "description": description}
        return func
    return decorator


@tool("read_file", "读取文件内容。输入: 文件路径（相对于 data/ 目录）。")
def read_file(path: str) -> str:
    p = (DATA_DIR / path.strip()).resolve()
    if not str(p).startswith(str(DATA_DIR.resolve())):
        return f"安全限制：只能读取 data/ 目录下的文件"
    if not p.exists():
        return f"文件不存在: {path}"
    try:
        content = p.read_text(encoding="utf-8")
        if len(content) > 80000:
            content = content[:80000] + f"\n... [文件过大，已截断到 80KB]"
        return content
    except Exception as e:
        return f"读取失败: {e}"


@tool("list_files", "列出 data/ 目录下的文件。输入: 忽略。")
def list_files(_: str) -> str:
    try:
        files = []
        for p in sorted(DATA_DIR.iterdir()):
            if p.is_file():
                size_kb = p.stat().st_size / 1024
                files.append(f"  {p.name} ({size_kb:.1f} KB)")
        return "data/ 目录文件:\n" + "\n".join(files) if files else "data/ 目录为空"
    except Exception as e:
        return f"列出失败: {e}"


@tool("calculate", "安全计算数学表达式。输入: 数学表达式字符串。")
def calculate(expr: str) -> str:
    ALLOWED_NODES = {
        ast.Expression, ast.Constant, ast.Name, ast.Load,
        ast.UnaryOp, ast.BinOp, ast.Add, ast.Sub, ast.Mult,
        ast.Div, ast.Pow, ast.FloorDiv, ast.Mod, ast.USub,
    }
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        return f"语法错误: {e}"
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            return f"不允许的语法: {type(node).__name__}"
    try:
        return str(eval(compile(tree, "<safe>", "eval"), {"__builtins__": {}}))
    except Exception as e:
        return f"计算错误: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 工具结果截断器
# ═══════════════════════════════════════════════════════════════════════════════

class ToolResultPruner:
    """工具输出截断器。

    对应 QwenPaw 的 _prune_tool_result():
      - 从消息列表末尾向前，识别连续的 Observation 消息
      - 预扫描 tool_use 收集豁免 tool_id
      - split_index 划分"最近区"和"旧消息区"
      - 最近 N 条：宽松截断（pruning_recent_msg_max_bytes）
      - 更早的：严格截断（pruning_old_msg_max_bytes）
      - 行完整性保护：在 \\n 边界切割
      - <<<TRUNCATED>>> 标记，LLM 可通过 read_file 按需读取完整内容
    """

    TRUNCATION_MARKER = "<<<TRUNCATED>>>"

    def __init__(
        self,
        config: PruningConfig | None = None,
        offload_dir: Path | None = None,
    ):
        self.config = config or PruningConfig()
        self.offload_dir = offload_dir or Path("tool_results_cache")
        self._offload_count = 0

    def prune(self, messages: list[dict]) -> list[dict]:
        """截断工具输出消息。异常安全：截断失败不影响对话流程。"""
        if not self.config.enabled or not messages:
            return messages

        try:
            return self._prune_impl(messages)
        except Exception as e:
            print(f"  [Pruner] 截断异常（跳过）: {e}")
            return messages

    def _prune_impl(self, messages: list[dict]) -> list[dict]:
        """截断实现。"""
        # Step 1: 找到连续的 Observation 消息索引
        tool_indices = self._find_tool_result_indices(messages)
        if not tool_indices:
            return messages

        # Step 2: 预扫描 tool_use 消息，收集豁免的 tool 名称
        exempt_tool_names = self._collect_exempt_tool_names(messages)

        # Step 3: 对每条 tool_result 应用差异化截断
        total = len(tool_indices)
        pruned_count = 0

        for pos, idx in enumerate(tool_indices):
            msg = messages[idx]
            tool_name = msg.get("tool_name", "")

            # 计算阈值：距离末尾越近越宽松
            distance_from_end = total - 1 - pos
            if tool_name in exempt_tool_names:
                # 豁免工具使用宽松阈值
                max_bytes = self.config.pruning_recent_msg_max_bytes
            elif distance_from_end < self.config.pruning_recent_n:
                max_bytes = self.config.pruning_recent_msg_max_bytes
            else:
                max_bytes = self.config.pruning_old_msg_max_bytes

            content = msg.get("content", "")
            if self._needs_truncation(content, max_bytes):
                truncated, _ = self._truncate_content(
                    content, max_bytes, idx
                )
                messages[idx] = {**msg, "content": truncated}
                pruned_count += 1

        if pruned_count:
            print(f"\n  [Pruner] 截断了 {pruned_count} 条工具输出")
        return messages

    # ── Step 1: 扫描 tool_result ────────────────────────────────────────────

    def _find_tool_result_indices(self, messages: list[dict]) -> list[int]:
        """从后向前找到连续的 Observation 消息索引。

        tool_use（^Action:）和 tool_result（Observation:）成对保留。
        末尾未执行的 tool_use 会被跳过。遇到非工具消息时中断扫描。
        """
        indices: list[int] = []
        i = len(messages) - 1

        if i >= 0 and self._is_tool_use(messages[i]):
            i -= 1

        while i >= 0:
            msg = messages[i]
            content = msg.get("content", "")
            if isinstance(content, str) and content.startswith("Observation:"):
                indices.append(i)
                i -= 1
                if i >= 0 and self._is_tool_use(messages[i]):
                    i -= 1
                continue
            break
        indices.reverse()
        return indices

    @staticmethod
    def _is_tool_use(msg: dict) -> bool:
        content = msg.get("content", "")
        return (
            msg.get("role") == "assistant"
            and isinstance(content, str)
            and bool(re.search(r"^Action:", content, re.MULTILINE))
        )

    # ── Step 2: 预扫描豁免 ─────────────────────────────────────────────────

    def _collect_exempt_tool_names(self, messages: list[dict]) -> set[str]:
        """预扫描所有 tool_use 消息，收集需要豁免的 tool 名称。

        对应 QwenPaw 的 Step 2: 收集 exempt_tool_ids：
          - tool_use 的 tool_name 在 exempt_tool_names 中 → 豁免
          - tool_use 是 read_file 且 raw_input 含豁免扩展名 → 豁免
        """
        if not self.config.exempt_tool_names and not self.config.exempt_file_extensions:
            return set()

        exempt: set[str] = set()
        for msg in messages:
            if not self._is_tool_use(msg):
                continue
            parsed = self._parse_tool_call(msg.get("content", ""))
            if not parsed:
                continue
            tname = parsed["action"]
            if tname in self.config.exempt_tool_names:
                exempt.add(tname)
            if tname == "read_file" and self.config.exempt_file_extensions:
                raw = parsed.get("action_input", "")
                if any(ext in raw for ext in self.config.exempt_file_extensions):
                    exempt.add(tname)
        return exempt

    @staticmethod
    def _parse_tool_call(content: str) -> dict | None:
        """从 tool_use 文本中提取 action 和 action_input。"""
        action_m = re.search(r"^Action:\s*(.+)$", content, re.MULTILINE)
        input_m = re.search(r"^Action Input:\s*(.+)$", content, re.MULTILINE)
        if not action_m:
            return None
        return {
            "action": action_m.group(1).strip(),
            "action_input": input_m.group(1).strip() if input_m else "",
        }

    # ── Step 3: 截断 ───────────────────────────────────────────────────────

    def _needs_truncation(self, content: str, max_bytes: int) -> bool:
        """检查内容是否需要截断（超过阈值 +100 字节容差）。"""
        return len(content.encode("utf-8")) > max_bytes + 100

    def _truncate_content(
        self, content: str, max_bytes: int, msg_index: int
    ) -> tuple[str, Path | None]:
        """截断内容，保持行完整性。

        对应 QwenPaw 的 truncate_text_output():
          - 在 max_bytes 字节边界处切割
          - 回退到最近的 \\n（保持行完整）
          - errors="ignore" 跳过断裂的多字节字符
          - 追加 <<<TRUNCATED>>> 通知
        """
        # 去掉 Observation: 前缀进行截断
        prefix = "Observation: "
        body = content[len(prefix):] if content.startswith(prefix) else content
        has_prefix = content.startswith(prefix)

        text_bytes = body.encode("utf-8")
        if len(text_bytes) <= max_bytes:
            return content, None

        # 在字节边界处切割
        truncated_bytes = text_bytes[:max_bytes]
        truncated_text = truncated_bytes.decode("utf-8", errors="ignore")

        # 回退到最后一个完整行（行完整性保护）
        last_newline = truncated_text.rfind("\n")
        if last_newline > 0:
            truncated_text = truncated_text[:last_newline]

        # 保存完整内容到文件
        offload_path = self._offload(body, msg_index)
        total_lines = body.count("\n") + 1
        kept_lines = truncated_text.count("\n") + 1

        # 构建截断通知（QwenPaw 格式）
        notice = (
            f"\n\n{self.TRUNCATION_MARKER}\n"
            f"The output above was truncated.\n"
            f"The full content is saved to the file and contains {total_lines} lines in total.\n"
            f"This excerpt starts at line 1 and covers the next {max_bytes} bytes ({kept_lines} lines).\n"
            f"If the current content is not enough, call `read_file` with "
            f"file_path={offload_path} start_line={kept_lines + 1} to read more.\n"
        )

        result = (prefix if has_prefix else "") + truncated_text + notice
        return result, offload_path

    def _offload(self, content: str, msg_index: int) -> Path:
        """将完整内容落盘，返回文件路径。"""
        self.offload_dir.mkdir(exist_ok=True)
        self._offload_count += 1
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = self.offload_dir / f"offload_{ts}_{msg_index:04d}.txt"
        fname.write_text(content, encoding="utf-8")
        return fname

    @property
    def stats(self) -> dict:
        return {"offload_count": self._offload_count, "offload_dir": str(self.offload_dir)}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ReAct Agent（对接 OpenAI）
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""\
你是一个 ReAct 智能体，通过思考和调用工具来回答问题。

可用工具：
{tool_descriptions}

每一步必须按以下格式输出：

Thought: <你对当前状态的推理，以及下一步计划>
Action: <工具名称>
Action Input: <传给工具的输入字符串>

获得足够信息后输出：

Thought: <对已获信息的总结>
Final Answer: <最终答案>

规则：
- 可以按顺序使用多个工具
- Observation 来自工具输出，用它来支撑推理
- 绝不要编造 Observation
""")


def _build_system_prompt() -> str:
    descriptions = "\n".join(
        f"  {name}: {info['description']}" for name, info in _tools.items()
    )
    return SYSTEM_PROMPT.format(tool_descriptions=descriptions)


def _parse_step(text: str) -> dict:
    result: dict = {"thought": "", "action": None, "action_input": None, "final_answer": None}

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
    if action not in _tools:
        return f"未知工具「{action}」。可用: {', '.join(_tools.keys())}"
    try:
        return _tools[action]["fn"](action_input)
    except Exception as e:
        return f"执行出错: {e}"


def run_react_conversation(question: str, model: str | None = None) -> list[dict]:
    """运行 ReAct 循环，返回完整消息历史（包含所有工具调用结果）。"""
    model = model or os.getenv("MODEL") or "gpt-4o-mini"
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    messages: list[dict] = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": question},
    ]

    print(f"\n{'=' * 60}")
    print(f"问题: {question}")
    print(f"{'=' * 60}\n")

    for step in range(1, MAX_STEPS + 1):
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=0.2,
        )
        text = response.choices[0].message.content or ""

        print(f"─── 第 {step} 步 ───")
        print(text)

        parsed = _parse_step(text)

        if parsed["final_answer"]:
            messages.append({"role": "assistant", "content": text})
            print(f"\n{'=' * 60}")
            print(f"答: {parsed['final_answer']}")
            print(f"{'=' * 60}")
            return messages

        if parsed["action"] and parsed["action_input"] is not None:
            observation = _execute_tool(parsed["action"], parsed["action_input"])
            print(f"  => Observation: ({len(observation)} 字符)\n")

            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}",
                "tool_name": parsed["action"],
            })
        else:
            print("  (未识别到 Action/Final Answer，结束)")
            break

    print("达到最大步数。")
    return messages


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 演示
# ═══════════════════════════════════════════════════════════════════════════════

def _show_messages(messages: list[dict], title: str) -> None:
    """展示消息列表概览。"""
    print(f"\n{title}:")
    total_bytes = 0
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        tool_name = msg.get("tool_name", "")
        size = len(content.encode("utf-8"))

        if isinstance(content, str) and content.startswith("Observation:"):
            truncated = "已截断" in content
            tag = " [截断]" if truncated else ""
            total_bytes += size
            label = f"  [{i}] Observation ({tool_name}, {size/1024:.0f}KB{tag})"
        elif role == "assistant" and ToolResultPruner._is_tool_use(msg):
            label = f"  [{i}] tool_use ({size} 字节)"
            total_bytes += size
        else:
            label = f"  [{i}] {role} ({size} 字节)"
            total_bytes += size
        print(label)
    print(f"  总大小: {total_bytes/1024:.0f} KB\n")


# ── 消息日志保存 ──────────────────────────────────────────────────────────────

def _make_user_view(messages: list[dict]) -> list[dict]:
    """将后端 messages 转换为用户视角的消息。

    转换规则：
      - system 消息 → 隐藏
      - user 消息（原始问题）→ 保留
      - tool_use（assistant + Action:）→ 替换为工具调用状态提示
      - Observation（user + Observation: 前缀）→ 替换为工具执行摘要
      - Final Answer（assistant 不含 Action:）→ 保留
    """
    result: list[dict] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            continue  # 用户看不到 system prompt

        if role == "user" and isinstance(content, str) and content.startswith("Observation:"):
            tool_name = msg.get("tool_name", "?")
            is_truncated = "<<<TRUNCATED>>>" in content
            size = len(content.encode("utf-8"))
            summary = (
                f"[工具调用完成] {tool_name}\n"
                f"返回内容大小: {size / 1024:.1f} KB"
            )
            if is_truncated:
                summary += "\n(内容已截断，LLM 可通过 read_file 按需读取完整内容)"
            result.append({"role": "tool_status", "content": summary, "tool_name": tool_name})
            continue

        if role == "assistant" and ToolResultPruner._is_tool_use(msg):
            parsed = ToolResultPruner._parse_tool_call(content)
            tool_name = parsed["action"] if parsed else "?"
            result.append({
                "role": "tool_status",
                "content": f"[正在调用工具] {tool_name}...",
                "tool_name": tool_name,
            })
            continue

        result.append({"role": role, "content": content})
    return result


def save_pruning_logs(backend_msgs: list[dict], pruner_stats: dict, tag: str = "") -> None:
    """保存后端消息和用户视角消息到 conversation_log/，方便对比学习。

    生成三个文件：
      - pruning_backend_{tag}_{timestamp}.json  — 发给 LLM 的完整消息
      - pruning_user_view_{tag}_{timestamp}.json — 用户看到的消息
      - pruning_comparison_{tag}_{timestamp}.json — 逐条对比
    """
    log_dir = Path(__file__).parent / "conversation_log"
    log_dir.mkdir(exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_prefix = f"pruning_{tag}_" if tag else "pruning_"

    user_view = _make_user_view(backend_msgs)

    # 1. 后端消息
    backend_path = log_dir / f"{tag_prefix}backend_{ts}.json"
    with open(backend_path, "w", encoding="utf-8") as f:
        json.dump({
            "description": "发给 LLM 的完整消息列表（包含 Observation 原始内容、截断标记等）",
            "pruner_stats": pruner_stats,
            "messages": backend_msgs,
        }, f, ensure_ascii=False, indent=2)

    # 2. 用户视角
    user_path = log_dir / f"{tag_prefix}user_view_{ts}.json"
    with open(user_path, "w", encoding="utf-8") as f:
        json.dump({
            "description": "用户看到的消息列表（推理过程隐藏，工具调用替换为状态提示）",
            "pruner_stats": pruner_stats,
            "messages": user_view,
        }, f, ensure_ascii=False, indent=2)

    # 3. 逐条对比
    comparison: list[dict] = []
    for i, backend_msg in enumerate(backend_msgs):
        role = backend_msg.get("role", "?")
        content = backend_msg.get("content", "")
        backend_summary = content[:200] + ("..." if len(content) > 200 else "")

        # 找到对应的用户视角消息
        user_match = _make_user_view([backend_msg])
        user_summary = user_match[0]["content"][:200] if user_match else "(隐藏)"

        comparison.append({
            "index": i,
            "backend": {
                "role": role,
                "tool_name": backend_msg.get("tool_name", ""),
                "preview": backend_summary,
                "size_bytes": len(content.encode("utf-8")),
            },
            "user_view": {
                "role": user_match[0]["role"] if user_match else "hidden",
                "preview": user_summary,
            },
        })

    comp_path = log_dir / f"{tag_prefix}comparison_{ts}.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump({
            "description": "逐条对比：后端发给 LLM 的消息 vs 用户视角看到的消息",
            "legend": {
                "backend": "发给 LLM 的消息（包含完整 Observation、截断标记、Thought 推理过程）",
                "user_view": "用户看到的消息（system 隐藏、工具调用显示为状态提示、Observation 不展示原始内容）",
            },
            "comparison": comparison,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n  日志已保存:")
    print(f"    后端消息:   {backend_path.name}")
    print(f"    用户视角:   {user_path.name}")
    print(f"    逐条对比:   {comp_path.name}")


def demo_with_ai():
    """主演示：运行真实 ReAct 对话，展示截断效果。"""
    # 如果没有配置 API key，使用模拟数据演示
    if not os.getenv("OPENAI_API_KEY"):
        print("未检测到 OPENAI_API_KEY，使用模拟数据演示。\n")
        demo_simulated()
        return

    question = (
        "请帮我分析 data/ 目录下的日志文件。"
        "先列出所有文件，然后读取 system_log.txt 和 large_log.txt，"
        "汇总其中的 ERROR 和 WARN 信息。"
    )

    print("\n" + "=" * 60)
    print("  演示: 真实 ReAct 对话 + 工具结果截断")
    print("=" * 60)
    print(f"\n使用模型: {os.getenv('MODEL', 'gpt-4o-mini')}")

    # 阶段 1: 运行 ReAct 对话
    messages = run_react_conversation(question)

    # 提取 Final Answer 之前的消息（模拟 post_acting 阶段）
    # 截断在对话进行中执行，此时最后一条是 Observation，下一个 reasoning 还没发生
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") == "assistant" and not ToolResultPruner._is_tool_use(m):
            last_assistant_idx = i

    if last_assistant_idx is None:
        # 没有纯文本 assistant 回复，使用全部消息
        pre_final = messages
    else:
        # 去掉最后的 Final Answer，模拟 post_acting 时的消息状态
        pre_final = messages[:last_assistant_idx]

    tool_msgs = [m for m in pre_final if m.get("content", "").startswith("Observation:")]
    if len(tool_msgs) >= 1:
        _show_messages(pre_final, "阶段 1: 截断前（模拟 post_acting 时的消息状态）")

        # 阶段 2: 应用截断
        print("─" * 60)
        print("阶段 2: 应用 ToolResultPruner")
        print("─" * 60)
        print(f"  配置: recent_n={PRUNING_RECENT_N}, "
              f"old_max={PRUNING_OLD_MAX_BYTES}B, "
              f"recent_max={PRUNING_RECENT_MAX_BYTES}B")

        pruner = ToolResultPruner()
        pruner.prune(pre_final)

        _show_messages(pre_final, "阶段 3: 截断后")
        print(f"统计: {json.dumps(pruner.stats, ensure_ascii=False)}")

        # 保存日志到 conversation_log/
        save_pruning_logs(pre_final, pruner.stats, tag="ai")
    else:
        print("\nAI 没有调用工具，无截断演示。")


def demo_simulated():
    """模拟演示：不调用 API，展示截断逻辑。"""
    messages = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": "分析 large_log.txt 中的错误信息"},
        {"role": "assistant", "content": "Action: list_files\nAction Input: /"},
        {
            "role": "user",
            "content": "Observation: data/ 目录文件:\n  large_log.txt (62.8 KB)\n  system_log.txt (2.1 KB)\n  weather_report.txt (1.2 KB)",
            "tool_name": "list_files",
        },
        {"role": "assistant", "content": "Action: read_file\nAction Input: system_log.txt"},
        {
            "role": "user",
            "content": f"Observation: {Path('data/system_log.txt').read_text('utf-8')}",
            "tool_name": "read_file",
        },
        {"role": "assistant", "content": "Action: read_file\nAction Input: large_log.txt"},
        {
            "role": "user",
            "content": f"Observation: {Path('data/large_log.txt').read_text('utf-8')}",
            "tool_name": "read_file",
        },
        {"role": "assistant", "content": "Action: read_file\nAction Input: weather_report.txt"},
        {
            "role": "user",
            "content": f"Observation: {Path('data/weather_report.txt').read_text('utf-8')}",
            "tool_name": "read_file",
        },
    ]

    _show_messages(messages, "截断前")

    print("─" * 60)
    print("应用 ToolResultPruner")
    print(f"配置: recent_n={PRUNING_RECENT_N}, "
          f"old_max={PRUNING_OLD_MAX_BYTES}B, "
          f"recent_max={PRUNING_RECENT_MAX_BYTES}B")
    print("─" * 60)

    pruner = ToolResultPruner()
    pruner.prune(messages)

    _show_messages(messages, "截断后")
    print(f"统计: {json.dumps(pruner.stats, ensure_ascii=False)}")

    # 保存日志到 conversation_log/
    save_pruning_logs(messages, pruner.stats, tag="sim")

    # 展示落盘文件
    offload_dir = Path("tool_results_cache")
    if offload_dir.exists():
        files = list(offload_dir.iterdir())
        if files:
            print(f"\n落盘文件 ({offload_dir}/):")
            for f in sorted(files):
                size_kb = f.stat().st_size / 1024
                print(f"  {f.name} ({size_kb:.0f} KB)")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("  QwenPaw Layer 0: 工具结果截断")
    print("  ================================")
    print()
    print("核心机制:")
    print("  1. 差异化截断 — 最近 N 条宽松(50KB)，更早严格(3KB)")
    print("  2. 完整落盘 — 截断内容写入 tool_results_cache/")
    print("  3. 工具调用成对保留 — 截断时 tool_use/tool_result 联动")
    print()
    print(f"测试数据: data/ 目录 ({len(list(DATA_DIR.iterdir()))} 个文件)")
    for p in sorted(DATA_DIR.iterdir()):
        if p.is_file():
            print(f"  - {p.name} ({p.stat().st_size/1024:.0f} KB)")

    demo_with_ai()

    # 清理
    import shutil
    cache_dir = Path("tool_results_cache")
    if cache_dir.exists():
        keep = input(f"\n保留 {cache_dir}/ 落盘文件? [Y/n]: ").strip().lower()
        if keep == "n":
            shutil.rmtree(cache_dir)
            print("已清理。")


if __name__ == "__main__":
    main()
