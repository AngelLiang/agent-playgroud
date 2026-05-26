"""
任务模式（Mission Mode）—— QwenPaw 最小化学习实现

对照 MISSION_MODE_PRD_DESIGN_zh.md 的核心概念：
  1. 文件驱动状态机 — 状态持久化在磁盘 JSON，进程重启不丢失
  2. Phase 1 / Phase 2 分离 — 规划（PM 角色）→ 执行（Developer 角色）
  3. 代码级循环控制 — 引擎读 prd.json 驱动迭代，不依赖 agent 自行判断
  4. Engine 是真相源 — 每轮迭代后从磁盘重读 prd.json，不相信 Final Answer
  5. Controller 模式 — Master 分派 Worker（简化为单 agent 模拟）
"""

from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import textwrap
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# 0. 配置
# ═══════════════════════════════════════════════════════════════════════════════

MISSIONS_DIR = Path(__file__).parent / "missions"
MAX_ITERATIONS = 5                # Phase 2 最大迭代次数
MAX_STEPS_PER_ITERATION = 10      # 每次 ReAct 循环最大步数
MAX_PRD_FIX_ATTEMPTS = 2          # PRD 自动修复最大尝试次数

# 工作区目录（由 MissionEngine 在初始化时设置）
_mission_workspace: Path | None = None


def _set_workspace(path: Path) -> None:
    global _mission_workspace
    _mission_workspace = path.resolve()


def _resolve_path(rel_path: str) -> Path:
    """在 workspace 内解析相对路径，防止路径穿越。"""
    if _mission_workspace is None:
        raise ValueError("workspace 未设置")
    p = (_mission_workspace / rel_path.strip()).resolve()
    if not str(p).startswith(str(_mission_workspace)):
        raise ValueError(f"路径穿越检测: {rel_path}")
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 工具注册表 + 内置工具
# ═══════════════════════════════════════════════════════════════════════════════

_tools: dict[str, dict] = {}


def tool(name: str, description: str):
    def decorator(func):
        _tools[name] = {"fn": func, "description": description}
        return func
    return decorator


@tool("list_files", "列出目录下的文件。输入: 相对于 workspace 的路径（如 '.' 表示根目录）。")
def list_files(path: str) -> str:
    try:
        p = _resolve_path(path.strip() or ".")
        if not p.exists():
            return f"路径不存在: {path}"
        if not p.is_dir():
            return f"不是目录: {path}"
        lines = []
        for item in sorted(p.iterdir()):
            tag = "[D]" if item.is_dir() else "[F]"
            size = f" ({item.stat().st_size / 1024:.1f} KB)" if item.is_file() else ""
            lines.append(f"  {tag} {item.name}{size}")
        return f"目录 {path or '.'} 内容:\n" + "\n".join(lines) if lines else f"目录 {path or '.'} 为空"
    except ValueError as e:
        return f"路径错误: {e}"
    except Exception as e:
        return f"列出失败: {e}"


@tool("read_file", "读取文件内容。输入: 相对于 workspace 的文件路径。")
def read_file(path: str) -> str:
    try:
        p = _resolve_path(path.strip())
        if not p.exists():
            return f"文件不存在: {path}"
        if not p.is_file():
            return f"不是文件: {path}"
        content = p.read_text(encoding="utf-8")
        if len(content) > 50000:
            content = content[:50000] + "\n... [文件过大，已截断到 50KB]"
        return content
    except ValueError as e:
        return f"路径错误: {e}"
    except Exception as e:
        return f"读取失败: {e}"


@tool("write_file", "写入文件内容。输入: JSON 字符串 {\"path\": \"相对路径\", \"content\": \"内容\"}。"
      "会自动创建父目录。")
def write_file(json_input: str) -> str:
    try:
        data = json.loads(json_input.strip())
        if not isinstance(data, dict) or "path" not in data or "content" not in data:
            return "参数错误：需要 JSON 格式 {\"path\": \"...\", \"content\": \"...\"}"
        p = _resolve_path(data["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data["content"], encoding="utf-8")
        size = len(data["content"].encode("utf-8"))
        return f"已写入: {data['path']} ({size} 字节)"
    except json.JSONDecodeError as e:
        return f"JSON 解析失败: {e}"
    except ValueError as e:
        return f"路径错误: {e}"
    except Exception as e:
        return f"写入失败: {e}"


@tool("execute_command", "在 workspace 目录下执行 shell 命令。输入: shell 命令字符串。30 秒超时。")
def execute_command(cmd: str) -> str:
    try:
        result = subprocess.run(
            cmd.strip(), shell=True, capture_output=True, text=True,
            timeout=30, cwd=str(_mission_workspace),
        )
        output = result.stdout.strip()
        if result.stderr.strip():
            output += ("\n" if output else "") + "[stderr]\n" + result.stderr.strip()
        return output or f"(exit code: {result.returncode})"
    except subprocess.TimeoutExpired:
        return "命令超时（30 秒）"
    except Exception as e:
        return f"执行失败: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 任务状态管理器
# ═══════════════════════════════════════════════════════════════════════════════

class MissionStateManager:
    """管理 Mission 状态文件系统。

    目录结构：
      missions/mission-{timestamp}/
        loop_config.json  — 阶段状态机
        prd.json          — 用户故事列表（含 passes 字段）
        progress.txt      — 追加式迭代日志
        task.md           — 原始任务描述

    Agent 的工作目录就是 mission 目录本身，所有文件操作都在此目录下进行。
    """

    def __init__(self, mission_dir: Path):
        self.mission_dir = mission_dir
        self.prd_path = mission_dir / "prd.json"
        self.config_path = mission_dir / "loop_config.json"
        self.progress_path = mission_dir / "progress.txt"
        self.task_path = mission_dir / "task.md"

    def init_mission(self, task: str, max_iterations: int = MAX_ITERATIONS) -> None:
        """创建 Mission 目录并初始化状态文件。"""
        self.mission_dir.mkdir(parents=True, exist_ok=True)

        self.task_path.write_text(task, encoding="utf-8")
        self.config_path.write_text(json.dumps({
            "mission_id": self.mission_dir.name,
            "phase": "prd_generation",
            "max_iterations": max_iterations,
            "current_iteration": 0,
            "task": task,
            "mission_dir": str(self.mission_dir),
            "created_at": datetime.datetime.now().isoformat(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        self.append_progress(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] Mission 创建, phase=prd_generation")

    # ── loop_config ──────────────────────────────────────────────────────────

    def read_loop_config(self) -> dict:
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    def write_loop_config(self, config: dict) -> None:
        self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    def advance_phase(self, new_phase: str) -> None:
        cfg = self.read_loop_config()
        old = cfg["phase"]
        cfg["phase"] = new_phase
        self.write_loop_config(cfg)
        self.append_progress(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] Phase 转换: {old} → {new_phase}")

    # ── prd.json ─────────────────────────────────────────────────────────────

    def read_prd(self) -> dict | None:
        if not self.prd_path.exists():
            return None
        try:
            return json.loads(self.prd_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, Exception):
            return None

    def write_prd(self, prd: dict) -> None:
        self.prd_path.write_text(json.dumps(prd, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_incomplete_stories(self) -> list[dict]:
        prd = self.read_prd()
        if not prd:
            return []
        return [s for s in prd.get("userStories", []) if not s.get("passes")]

    # ── progress.txt ─────────────────────────────────────────────────────────

    def append_progress(self, entry: str) -> None:
        with open(self.progress_path, "a", encoding="utf-8") as f:
            f.write(entry + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PRD 校验
# ═══════════════════════════════════════════════════════════════════════════════

_REQUIRED_STORY_FIELDS = {"id", "title", "description", "acceptanceCriteria"}


def _validate_prd(prd: dict) -> list[str]:
    """验证 prd.json Schema。返回问题列表，空列表 = 有效。自动修复缺失的 passes/priority。"""
    problems: list[str] = []

    if not isinstance(prd, dict):
        return ["prd.json 必须是 JSON 对象"]
    if "userStories" not in prd:
        return ["缺少 userStories 字段"]
    stories = prd.get("userStories", [])
    if not isinstance(stories, list) or len(stories) == 0:
        return ["userStories 必须是非空数组"]

    for i, s in enumerate(stories):
        if not isinstance(s, dict):
            problems.append(f"userStories[{i}]: 必须是对象")
            continue
        for field in _REQUIRED_STORY_FIELDS:
            if field not in s:
                problems.append(f"userStories[{i}]: 缺少 '{field}'")
        if "acceptanceCriteria" in s and not isinstance(s["acceptanceCriteria"], list):
            problems.append(f"userStories[{i}]: acceptanceCriteria 必须是数组")
        # 自动修复缺失的 passes / priority
        if "passes" not in s:
            s["passes"] = False
        if "priority" not in s:
            s["priority"] = "P2"

    return problems


def _print_prd_summary(prd: dict) -> None:
    """打印 PRD 摘要供用户确认。"""
    stories = prd.get("userStories", [])
    print(f"\n{'─' * 60}")
    print(f"PRD 摘要 — {len(stories)} 个用户故事:")
    print(f"{'─' * 60}")
    for s in stories:
        status = "✓" if s.get("passes") else "⬜"
        print(f"  {status} {s.get('id', '?')}: {s.get('title', '?')}")
        if s.get("description"):
            print(f"     {s['description'][:100]}")
    print(f"{'─' * 60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ReAct 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def _build_system_prompt(template: str) -> str:
    descriptions = "\n".join(
        f"  {name}: {info['description']}" for name, info in _tools.items()
    )
    return template.format(tool_descriptions=descriptions)


def _parse_step(text: str) -> dict:
    result: dict = {"thought": "", "action": None, "action_input": None, "final_answer": None}

    def _get(key: str) -> str | None:
        # 兼容纯文本和 Markdown 格式: "Final Answer:" / "**Final Answer:**" / "**Final Answer**"
        pattern = rf"^(?:\*\*)?{re.escape(key)}(?:\*\*)?:\s*(.*?)$"
        m = re.search(pattern, text, re.MULTILINE | re.DOTALL)
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
        available = ", ".join(_tools.keys())
        return f"未知工具「{action}」。可用: {available}"
    try:
        return _tools[action]["fn"](action_input)
    except Exception as e:
        return f"执行「{action}」时出错: {e}"


def _save_conversation_log(messages: list[dict], tag: str, meta: dict | None = None) -> Path:
    log_dir = Path(__file__).parent / "conversation_log"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"mission_{tag}_{ts}.json"
    data = {"tag": tag, "timestamp": ts, "messages": messages}
    if meta:
        data["meta"] = meta
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Phase System Prompts
# ═══════════════════════════════════════════════════════════════════════════════

PHASE_1_SYSTEM_PROMPT = textwrap.dedent("""\
你是 Mission Mode 的 Phase 1 产品经理（PM）。
你的任务是将用户需求分解为结构化的用户故事（User Stories），输出 prd.json。

## 可用工具
{tool_descriptions}

## 输出格式（每一步必须严格遵循，不要使用 Markdown 格式）

Thought: <你对当前状态的推理>
Action: <工具名称>
Action Input: <工具输入>

完成后输出：

Thought: <总结 PRD 内容>
Final Answer: <PRD 摘要>

注意：Final Answer 前面没有 ## 或 **，就是纯文本 "Final Answer:"。

## Phase 1 工作流程（按顺序执行）

1. 使用 list_files 查看当前目录
2. 使用 read_file 读取 task.md
3. 生成 3-5 个用户故事
4. **先使用 write_file 写入 prd.json**，然后才能输出 Final Answer
5. write_file 的输入格式:
   {{"path": "prd.json", "content": "<完整 JSON>"}}
6. prd.json 格式:
   {{
     "project": "项目名称",
     "description": "项目简述",
     "userStories": [
       {{
         "id": "US-001",
         "title": "...",
         "description": "As a user, I want ...",
         "acceptanceCriteria": ["条件1", "条件2"],
         "priority": "P1",
         "passes": false
       }}
     ]
   }}

## 规则
- **必须先用 write_file 写入 prd.json，再输出 Final Answer**
- 将任务分解为 3-5 个用户故事
- 每个故事可独立实现、可验证
""")

PHASE_2_SYSTEM_PROMPT = textwrap.dedent("""\
你是 Mission Mode 的 Phase 2 开发者。
你的任务是根据 prd.json 中的用户故事，逐个实现并更新完成状态。

## 可用工具
{tool_descriptions}

## 输出格式（每一步必须遵循）

Thought: <你对当前状态的推理>
Action: <工具名称>
Action Input: <工具输入>

## Phase 2 工作流程

1. 使用 read_file 读取 prd.json，了解有哪些故事需要实现
2. 对于每个未完成的故事（passes=false）:
   a. 使用 write_file 创建/修改代码文件
   b. 使用 execute_command 运行测试验证
   c. 使用 write_file 更新 prd.json，将该故事的 passes 设为 true
      **重要**: 保持其他字段不变，只修改 passes
3. 使用 write_file 更新 progress.txt

## 规则
- 每次迭代聚焦 1-2 个故事
- **修改 prd.json 时必须保持其余故事完整**——先 read_file 读取当前 prd.json
- 完成后输出 Final Answer 总结本次迭代的进展
""")

PRD_FIX_PROMPT = textwrap.dedent("""\
你之前生成的 prd.json 存在以下问题：

{problems}

请使用 write_file 重新写入修正后的 prd.json。确保每个用户故事都包含 id, title, description, acceptanceCriteria, priority, passes 字段。
""")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MissionEngine 主控制器
# ═══════════════════════════════════════════════════════════════════════════════

class MissionEngine:
    """Mission Mode 执行引擎。

    对照 QwenPaw 的 mission_runner.py:
      - run() = mission_runner.run_mission_phase1/2()
      - _phase_1_prd_generation() = Phase 1: PRD 生成 + 校验 + 用户确认
      - _phase_2_execution() = Phase 2: 代码级循环，Engine 读 prd.json 驱动迭代
    """

    def __init__(self, task: str, max_iterations: int = MAX_ITERATIONS, model: str | None = None):
        self.task = task
        self.max_iterations = max_iterations
        self.model = model or os.getenv("MODEL") or "gpt-4o-mini"

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("未设置 OPENAI_API_KEY，请检查 .env 文件")

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

        # 创建 Mission 目录
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.mission_dir = MISSIONS_DIR / f"mission-{ts}"
        self.state = MissionStateManager(self.mission_dir)
        self.state.init_mission(task, max_iterations)

        # 设置全局 workspace = mission 目录（agent 工作目录即 mission 目录）
        _set_workspace(self.state.mission_dir)

    def run(self) -> None:
        """主入口：Phase 1 → 用户确认 → Phase 2 → 终态。"""
        print(f"\n{'=' * 60}")
        print(f"  Mission Mode — 任务分解与执行")
        print(f"{'=' * 60}")
        print(f"  Mission: {self.mission_dir.name}")
        print(f"  任务: {self.task[:80]}{'...' if len(self.task) > 80 else ''}")
        print(f"  最大迭代: {self.max_iterations}")
        print(f"{'=' * 60}")

        try:
            # Phase 1: PRD 生成
            confirmed = self._phase_1_prd_generation()
            if not confirmed:
                print("\n用户取消，Mission 终止。")
                return

            # Phase 2: 执行循环
            self._phase_2_execution()
        except KeyboardInterrupt:
            print("\n\n用户中断。")
            self._print_final_status()
        except Exception as e:
            print(f"\nMission 异常: {e}")
            self._print_final_status()
            raise

    # ── Phase 1 ──────────────────────────────────────────────────────────────

    def _phase_1_prd_generation(self) -> bool:
        """Phase 1: 生成 PRD。返回 True 表示用户确认。"""
        print(f"\n{'─' * 60}")
        print("  Phase 1: PRD 生成（PM 角色）")
        print(f"{'─' * 60}")

        system_prompt = _build_system_prompt(PHASE_1_SYSTEM_PROMPT)
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"任务描述: {self.task}\n\n"
                f"请按 Phase 1 工作流程执行:\n"
                f"1. 探索 workspace\n"
                f"2. 生成用户故事\n"
                f"3. 使用 write_file 写入 prd.json\n"
                f"4. 输出 Final Answer 总结 PRD"
            )},
        ]

        # 运行 agent（含自动修复循环）
        for attempt in range(1 + MAX_PRD_FIX_ATTEMPTS):
            print(f"\n  [尝试 {attempt + 1}] 运行 Agent...\n")
            messages = self._run_react_loop(messages, max_steps=MAX_STEPS_PER_ITERATION)

            # 检查 prd.json
            prd = self.state.read_prd()
            if prd is None:
                # 调试：列出目录内容
                print(f"\n  [调试] Engine 预期 prd.json 路径: {self.state.prd_path}")
                existing = list(self.mission_dir.rglob("*"))
                print(f"  [调试] Mission 目录现有文件 ({len(existing)} 个):")
                for p in sorted(existing):
                    print(f"    {'[D]' if p.is_dir() else '[F]'} {p.relative_to(self.mission_dir)}")
                if attempt < MAX_PRD_FIX_ATTEMPTS:
                    print("\n  ⚠️  未找到 prd.json，要求 agent 重新生成...")
                    messages.append({"role": "user", "content": "未找到 prd.json，请使用 write_file 将 prd.json 写入磁盘。确保 path 为 \"prd.json\"。"})
                    continue
                else:
                    print("\n  ❌ 多次尝试后仍未生成 prd.json")
                    return False

            # 校验 PRD
            problems = _validate_prd(prd)
            if problems:
                if attempt < MAX_PRD_FIX_ATTEMPTS:
                    print(f"\n  ⚠️  PRD 校验发现问题 ({len(problems)} 个):")
                    for p in problems:
                        print(f"    - {p}")
                    fix_msg = PRD_FIX_PROMPT.format(problems="\n".join(f"  - {p}" for p in problems))
                    messages.append({"role": "user", "content": fix_msg})
                    continue
                else:
                    print(f"\n  ❌ PRD 校验仍失败")
                    return False

            # 校验通过
            break

        # 显示 PRD 摘要
        _print_prd_summary(prd)
        self.state.advance_phase("execution_confirmed")

        # 用户确认
        confirm = input("是否开始执行 Phase 2? [Y/n]: ").strip().lower()
        if confirm and confirm not in ("y", "yes", "是"):
            return False

        self.state.advance_phase("execution")
        return True

    # ── Phase 2 ──────────────────────────────────────────────────────────────

    def _phase_2_execution(self) -> None:
        """Phase 2: 执行循环 — Engine 控制迭代，读取 prd.json 判断进度。"""
        print(f"\n{'─' * 60}")
        print("  Phase 2: 执行循环（Developer 角色）")
        print(f"{'─' * 60}")

        prd = self.state.read_prd()
        if not prd:
            print("  ❌ prd.json 丢失，无法进入 Phase 2")
            return

        total_stories = len(prd.get("userStories", []))

        # 构建 PRD 摘要作为 Phase 2 的初始上下文
        prd_summary_lines = ["当前 PRD 状态:", ""]
        for s in prd.get("userStories", []):
            status = "✓" if s.get("passes") else "⬜"
            prd_summary_lines.append(
                f"  {status} {s['id']}: {s['title']} "
                f"[{', '.join(s.get('acceptanceCriteria', []))}]"
            )
        prd_summary = "\n".join(prd_summary_lines)

        system_prompt = _build_system_prompt(PHASE_2_SYSTEM_PROMPT)
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"{prd_summary}\n\n"
                f"请开始实现用户故事。每次迭代聚焦 1-2 个故事，完成后更新 prd.json 中的 passes 字段。"
            )},
        ]

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'─' * 40}")
            print(f"  迭代 {iteration}/{self.max_iterations}")
            print(f"{'─' * 40}")

            cfg = self.state.read_loop_config()
            cfg["current_iteration"] = iteration
            self.state.write_loop_config(cfg)

            # Engine 从磁盘读取 prd.json（真相源）
            prd = self.state.read_prd()
            incomplete = self.state.get_incomplete_stories()
            passed = total_stories - len(incomplete)

            print(f"  Engine 检查 prd.json: {passed}/{total_stories} 故事已完成")

            if not incomplete:
                self.state.advance_phase("completed")
                print(f"\n  ✅ 所有故事已完成！")
                break

            # 发送继续消息
            remaining_list = "\n".join(
                f"  ⬜ {s['id']}: {s['title']}" for s in incomplete
            )
            continuation = (
                f"[Mission — 迭代 {iteration}/{self.max_iterations}] "
                f"{passed}/{total_stories} 故事已通过，{len(incomplete)} 个待完成:\n\n"
                f"{remaining_list}\n\n"
                f"请继续实现待完成的故事。完成后使用 write_file 更新 prd.json（只修改对应故事的 passes 字段，保持其他故事完整）。"
            )
            messages.append({"role": "user", "content": continuation})

            # 运行 agent
            messages = self._run_react_loop(messages, max_steps=MAX_STEPS_PER_ITERATION)

            # 保存本轮对话日志
            _save_conversation_log(messages, f"phase2_iter{iteration}", {
                "iteration": iteration, "phase": "execution",
            })

            # Engine 再次从磁盘读取 prd.json
            prd = self.state.read_prd()
            if prd:
                self.state.append_progress(
                    f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] "
                    f"迭代 {iteration} 结束, "
                    f"{sum(1 for s in prd.get('userStories', []) if s.get('passes'))}/{total_stories} 已完成"
                )

        else:
            # for 循环正常结束（未 break）→ 达到最大迭代
            self.state.advance_phase("max_iterations_reached")
            print(f"\n  ⚠️  达到最大迭代次数 ({self.max_iterations})")

        self._print_final_status()

    # ── ReAct 循环 ───────────────────────────────────────────────────────────

    def _run_react_loop(self, messages: list[dict], max_steps: int = MAX_STEPS_PER_ITERATION) -> list[dict]:
        """运行 ReAct 循环，返回更新后的消息列表。"""
        for step in range(1, max_steps + 1):
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.2,
            )
            text = response.choices[0].message.content or ""

            print(f"  [步骤 {step}] ", end="")
            # 只打印第一行关键信息
            first_line = text.split("\n")[0] if text else "(空)"
            print(first_line[:100])

            parsed = _parse_step(text)

            if parsed["final_answer"]:
                messages.append({"role": "assistant", "content": text})
                print(f"    => Final Answer: {parsed['final_answer'][:100]}")
                return messages

            if parsed["action"] and parsed["action_input"] is not None:
                observation = _execute_tool(parsed["action"], parsed["action_input"])
                obs_len = len(observation)
                print(f"    => Observation ({parsed['action']}, {obs_len} 字符)")

                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}",
                    "tool_name": parsed["action"],
                })
            else:
                print("    (未识别到 Action/Final Answer)")
                break

        return messages

    # ── 终态 ─────────────────────────────────────────────────────────────────

    def _print_final_status(self) -> None:
        """打印最终状态。"""
        print(f"\n{'=' * 60}")
        print(f"  Mission 终态")
        print(f"{'=' * 60}")

        cfg = self.state.read_loop_config()
        prd = self.state.read_prd()

        print(f"  Phase: {cfg.get('phase', '?')}")
        print(f"  迭代: {cfg.get('current_iteration', 0)}/{cfg.get('max_iterations', '?')}")

        if prd:
            stories = prd.get("userStories", [])
            passed = sum(1 for s in stories if s.get("passes"))
            print(f"  故事: {passed}/{len(stories)} 已完成")
            for s in stories:
                status = "✓" if s.get("passes") else "⬜"
                print(f"    {status} {s.get('id', '?')}: {s.get('title', '?')}")

        print(f"\n  Mission 目录: {self.mission_dir}")
        progress = self.state.progress_path.read_text(encoding="utf-8").strip()
        print(f"\n  进度日志:")
        for line in progress.split("\n")[-5:]:
            print(f"    {line}")

        _save_conversation_log([], f"final_{cfg.get('phase', 'unknown')}", {
            "phase": cfg.get("phase"),
            "prd": prd,
            "config": cfg,
        })


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CLI 入口
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_TASK = (
    "创建一个 Python 工具模块 utils.py，包含以下功能：\n"
    "1. 字符串处理函数（去除首尾空格、反转字符串、统计字符数）\n"
    "2. 数学工具函数（斐波那契数列、判断质数）\n"
    "3. 每个函数需要有 docstring 和基本的错误处理"
)


def main():
    import sys

    print()
    print("  QwenPaw Mission Mode — 任务分解与执行")
    print("  =========================================")
    print()
    print("核心机制:")
    print("  1. 文件驱动状态机 — 状态持久化在 missions/ 目录")
    print("  2. Phase 1: PRD 生成（PM 角色）→ 用户确认")
    print("  3. Phase 2: 执行循环（Developer 角色）→ Engine 读 prd.json 判停")
    print("  4. Engine 是真相源 — 每轮从磁盘重读，不信任 Final Answer")
    print()

    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        print("选择任务:")
        print(f"  1. 默认任务 — {DEFAULT_TASK.split(chr(10))[0]}")
        print("  2. 自定义任务")
        print()
        choice = input("请选择 [1-2]: ").strip()
        if choice == "2":
            print("请输入任务描述（输入 END 结束）:")
            lines = []
            while True:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            task = "\n".join(lines).strip() or DEFAULT_TASK
        else:
            task = DEFAULT_TASK

    try:
        engine = MissionEngine(task=task)
        engine.run()
    except RuntimeError as e:
        print(f"\n错误: {e}")
        print("请在 .env 文件中设置 OPENAI_API_KEY")


if __name__ == "__main__":
    main()
