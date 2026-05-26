"""
Claude Code 风格 Agent — 使用 OpenAI Function Calling 实现工具调用。

核心设计：
  1. Function Calling — 工具定义通过 tools 参数传入 API，模型返回结构化 tool_calls
  2. 工具集 — 模仿 Claude Code 的关键工具（grep, glob, read, write, edit, execute, task）
  3. 安全边界 — 文件操作限定在 workspace 内，命令执行有超时，禁止危险操作
  4. 上下文管理 — 大文件自动截断，工具输出合理裁剪

对比文本解析方案的优势：
  - 零解析: 模型返回结构化 JSON，不需要正则提取 Thought/Action
  - 格式可靠: 模型原生支持 tool_calls，不会出现"不换行导致解析失败"
  - 并行调用: 部分模型支持一次返回多个 tool_calls
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

MAX_STEPS = 15
READ_MAX_BYTES = 80_000          # read_file 最大字节数
GREP_MAX_MATCHES = 50            # grep 最多返回的匹配行数
GLOB_MAX_FILES = 100             # glob 最多返回的文件数
EXECUTE_TIMEOUT = 60             # 命令执行超时（秒）

_workspace: Path = Path.cwd()


def _set_workspace(path: Path) -> None:
    global _workspace
    _workspace = path.resolve()


def _resolve_path(rel_path: str) -> Path:
    """在 workspace 内解析相对路径，拒绝路径穿越。"""
    p = (_workspace / rel_path.strip()).resolve()
    if not str(p).startswith(str(_workspace)):
        raise ValueError(f"路径穿越检测: {rel_path}")
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 工具注册表
# ═══════════════════════════════════════════════════════════════════════════════

_tools: dict[str, dict] = {}


def tool(name: str, description: str):
    def decorator(func):
        _tools[name] = {"fn": func, "description": description}
        return func
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 内置工具 — 模仿 Claude Code 核心工具
# ═══════════════════════════════════════════════════════════════════════════════


@tool(
    "grep",
    "在文件中搜索匹配的文本模式。\n"
    "输入: JSON 字符串 {{\"pattern\": \"正则或字面文本\", \"path\": \"文件或目录(可选,默认.)\", "
    "\"include\": \"文件匹配glob(可选,如**/*.py)\"}}\n"
    "path 可以是文件或目录。include 用于过滤文件类型（如 **/*.py 递归匹配所有 Python 文件）。\n"
    "返回匹配的文件路径、行号及内容。\n"
    "提示: 搜索结果会按文件排序，达到上限后会列出被跳过的文件，方便你缩小范围继续搜索。\n"
    "示例: {{\"pattern\": \"def calculate\"}}  在整个项目搜索\n"
    "     {{\"pattern\": \"TODO\", \"include\": \"**/*.py\"}}  只搜 Python 文件\n"
    "     {{\"pattern\": \"class\", \"path\": \"src/main.py\"}}  搜单个文件",
)
def grep_tool(json_input: str) -> str:
    try:
        params = json.loads(json_input.strip())
        if not isinstance(params, dict):
            return "参数错误: 需要 JSON 对象"
    except json.JSONDecodeError:
        params = {"pattern": json_input.strip()}

    pattern = params.get("pattern", "")
    if not pattern:
        return "错误: 必须提供 pattern"

    search_path = _resolve_path(params.get("path", "."))
    include = params.get("include") or params.get("glob") or "*"

    if not search_path.exists():
        return f"路径不存在: {params.get('path', '.')}"

    try:
        regex = re.compile(pattern)
    except re.error:
        regex = re.compile(re.escape(pattern))

    # 收集待搜索的文件列表
    files_to_search: list[Path] = []
    if search_path.is_file():
        files_to_search = [search_path]
    elif search_path.is_dir():
        for fp in sorted(search_path.rglob(include)):
            rel = fp.relative_to(_workspace)
            parts = rel.parts
            if any(p.startswith(".") and p != "." for p in parts):
                continue
            if any(p in ("node_modules", "__pycache__", ".venv", "venv", ".git")
                   for p in parts):
                continue
            if fp.is_file():
                files_to_search.append(fp)
    else:
        return f"路径类型不支持: {params.get('path', '.')}"

    matches: list[str] = []
    searched_files: list[str] = []
    skipped_files: list[str] = []
    hit_limit = False

    for filepath in files_to_search:
        if hit_limit:
            skipped_files.append(str(filepath.relative_to(_workspace)))
            continue

        rel = str(filepath.relative_to(_workspace))
        file_matches: list[str] = []
        try:
            for lineno, line in enumerate(filepath.read_text(encoding="utf-8").split("\n"), 1):
                if regex.search(line):
                    display = line[:200]
                    file_matches.append(f"{rel}:{lineno}: {display}")
        except (UnicodeDecodeError, PermissionError, OSError):
            continue

        if file_matches:
            searched_files.append(rel)
            space_left = GREP_MAX_MATCHES - len(matches)
            if len(file_matches) <= space_left:
                matches.extend(file_matches)
            else:
                # 放得下多少放多少
                matches.extend(file_matches[:space_left])
                hit_limit = True
                # 记录当前文件剩余匹配
                remaining_in_file = len(file_matches) - space_left
                if remaining_in_file > 0:
                    skipped_files.append(f"{rel} (还有 {remaining_in_file}+ 条未显示)")
                # 记录后续未搜索的文件，然后退出循环
                idx = files_to_search.index(filepath)
                for fp in files_to_search[idx + 1:]:
                    skipped_files.append(str(fp.relative_to(_workspace)))
                break

    if not matches:
        return f"未找到匹配 '{pattern}' 的内容 (路径: {params.get('path', '.')}, 模式: {include})"

    result = "\n".join(matches)
    if hit_limit:
        result += f"\n\n── 已达上限 {GREP_MAX_MATCHES} 条，以下文件未搜索或未完整显示 ──"
        for sf in skipped_files[:15]:
            result += f"\n  ⬜ {sf}"
        if len(skipped_files) > 15:
            result += f"\n  ... 还有 {len(skipped_files) - 15} 个文件"
        result += (
            f"\n\n建议: 缩小搜索范围，例如指定 path 为某个子目录，"
            f"或使用更精确的 pattern 减少匹配数。"
        )
    return result


@tool(
    "glob",
    "按 glob 模式查找文件。\n"
    "输入: JSON 字符串 {{\"pattern\": \"**/*.py\", \"path\": \"起始目录(可选)\"}}\n"
    "或直接传 glob 模式字符串如 \"**/*.py\"。\n"
    "示例: {{\"pattern\": \"**/*.py\"}} 查找所有 Python 文件。",
)
def glob_tool(json_input: str) -> str:
    try:
        params = json.loads(json_input.strip())
        if not isinstance(params, dict):
            return "参数错误: 需要 JSON 对象"
    except json.JSONDecodeError:
        params = {"pattern": json_input.strip()}

    pattern = params.get("pattern", "")
    if not pattern:
        return "错误: 必须提供 pattern"

    search_dir = _resolve_path(params.get("path", "."))
    if not search_dir.exists():
        return f"路径不存在: {params.get('path', '.')}"

    results: list[str] = []
    for filepath in sorted(search_dir.rglob(pattern)):
        if len(results) >= GLOB_MAX_FILES:
            results.append(f"... (达到上限 {GLOB_MAX_FILES} 个文件, 已截断)")
            break
        rel = filepath.relative_to(_workspace)
        parts = rel.parts
        if any(p.startswith(".") and p != "." for p in parts):
            continue
        if any(p in ("node_modules", "__pycache__", ".venv", "venv", ".git")
               for p in parts):
            continue
        if filepath.is_file():
            results.append(str(rel))

    if not results:
        return f"未找到匹配 '{pattern}' 的文件 (目录: {params.get('path', '.')})"
    return "\n".join(results)


@tool(
    "read",
    "读取文件内容。\n"
    "输入: 相对于 workspace 的文件路径。\n"
    "大文件自动截断到 {READ_MAX_BYTES} 字节(约 {lines} 行)。\n"
    "示例: src/main.py 或 path/to/file.js",
)
def read_file_tool(path_str: str) -> str:
    try:
        p = _resolve_path(path_str.strip())
        if not p.exists():
            return f"文件不存在: {path_str}"
        if p.is_dir():
            # 列出目录内容
            items = sorted(p.iterdir())
            lines = []
            for item in items:
                tag = "D" if item.is_dir() else "F"
                size = f" ({item.stat().st_size / 1024:.1f}KB)" if item.is_file() else ""
                lines.append(f"  [{tag}] {item.name}{size}")
            return f"{path_str} 是目录:\n" + ("\n".join(lines) if lines else "(空)")
        content = p.read_text(encoding="utf-8")
        if len(content) > READ_MAX_BYTES:
            content = content[:READ_MAX_BYTES] + f"\n\n... [文件过大: {len(content)} 字节, 已截断到 {READ_MAX_BYTES} 字节]"
        return content
    except ValueError as e:
        return f"路径错误: {e}"
    except UnicodeDecodeError:
        return f"无法以 UTF-8 解码: {path_str} (可能是二进制文件)"
    except Exception as e:
        return f"读取失败: {e}"


@tool(
    "write",
    "创建或覆盖文件。\n"
    "输入: JSON 字符串 {{\"path\": \"相对路径\", \"content\": \"文件内容\"}}\n"
    "会自动创建父目录。注意: 这会覆盖已有文件！",
)
def write_file_tool(json_input: str) -> str:
    try:
        data = json.loads(json_input.strip())
        if not isinstance(data, dict) or "path" not in data or "content" not in data:
            return "参数错误: 需要 {\"path\": \"...\", \"content\": \"...\"}"
        p = _resolve_path(data["path"])
        existed = p.exists()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data["content"], encoding="utf-8")
        size = len(data["content"].encode("utf-8"))
        action = "已更新" if existed else "已创建"
        return f"{action}: {data['path']} ({size} 字节)"
    except json.JSONDecodeError as e:
        return f"JSON 解析失败: {e}"
    except ValueError as e:
        return f"路径错误: {e}"
    except Exception as e:
        return f"写入失败: {e}"


@tool(
    "edit",
    "在文件中进行精确字符串替换（类似 sed 的精确替换）。\n"
    "输入: JSON 字符串 {{\"path\": \"文件路径\", \"old_string\": \"要替换的原文本\", \"new_string\": \"替换后的文本\"}}\n"
    "要求 old_string 在文件中唯一出现（防止误修改），或者设置 \"replace_all\": true 替换所有出现。\n"
    "示例: {{\"path\": \"src/main.py\", \"old_string\": \"def old_name():\", \"new_string\": \"def new_name():\"}}",
)
def edit_file_tool(json_input: str) -> str:
    try:
        data = json.loads(json_input.strip())
    except json.JSONDecodeError as e:
        return f"JSON 解析失败: {e}"

    if not isinstance(data, dict):
        return "参数错误: 需要 JSON 对象"

    file_path = data.get("path", "")
    old = data.get("old_string", "")
    new = data.get("new_string", "")
    replace_all = data.get("replace_all", False)

    if not file_path or not old:
        return "错误: 必须提供 path 和 old_string"

    try:
        p = _resolve_path(file_path.strip())
    except ValueError as e:
        return f"路径错误: {e}"

    if not p.exists():
        return f"文件不存在: {file_path}"
    if not p.is_file():
        return f"不是文件: {file_path}"

    try:
        content = p.read_text(encoding="utf-8")
    except Exception as e:
        return f"读取文件失败: {e}"

    count = content.count(old)
    if count == 0:
        return f"未找到匹配的 old_string (文件: {file_path})\n提示: 检查缩进和特殊字符是否完全一致"
    if not replace_all and count > 1:
        return (
            f"old_string 出现了 {count} 次 (文件: {file_path})。\n"
            f"请提供更多上下文使 old_string 唯一, 或设置 replace_all: true 以替换全部。"
        )

    new_content = content.replace(old, new)
    p.write_text(new_content, encoding="utf-8")
    return f"已编辑: {file_path} ({count} 处替换)"


@tool(
    "execute",
    "在 workspace 目录下执行 shell 命令。\n"
    "输入: shell 命令字符串。\n"
    f"超时: {EXECUTE_TIMEOUT} 秒。返回 stdout 和 stderr。\n"
    "⚠️ 禁止执行交互式命令或破坏性命令 (rm -rf, git push --force 等需确认)。",
)
def execute_tool(cmd: str) -> str:
    cmd = cmd.strip()
    # 危险命令检查
    dangerous_patterns = [
        (r"\brm\s+-rf?\b", "rm -rf 是破坏性命令"),
        (r"\bgit\s+push\s+.*--force", "git push --force 可能覆盖上游"),
        (r"\bgit\s+reset\s+--hard\b", "git reset --hard 会丢失本地修改"),
        (r"\bsudo\b", "sudo 需要额外确认"),
        (r">\s*/dev/", "输出重定向到设备文件"),
    ]
    for pattern, warning in dangerous_patterns:
        if re.search(pattern, cmd):
            return f"⚠️ 命令被拒绝: {warning}\n如需执行, 请手动在终端运行。"

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=EXECUTE_TIMEOUT, cwd=str(_workspace),
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        parts = []
        if out:
            parts.append(out[:10000])
        if err:
            parts.append(f"[stderr]\n{err[:5000]}")
        return "\n".join(parts) or f"(exit code: {result.returncode})"
    except subprocess.TimeoutExpired:
        return f"命令超时 ({EXECUTE_TIMEOUT} 秒)"
    except Exception as e:
        return f"执行失败: {e}"


@tool(
    "task",
    "管理任务清单（Todo 列表），用于跟踪进度。\n"
    "输入: JSON 字符串 {{\"action\": \"list|add|done\", "
    "\"content\": \"任务描述(add时必需)\", \"index\": 任务序号(done时必需)}}\n"
    "示例: {{\"action\": \"add\", \"content\": \"修复登录 bug\"}}\n"
    "      {{\"action\": \"list\"}}\n"
    "      {{\"action\": \"done\", \"index\": 1}}",
)
def task_tool(json_input: str) -> str:
    try:
        data = json.loads(json_input.strip())
    except json.JSONDecodeError:
        return "参数错误: 需要 JSON 对象"

    if not isinstance(data, dict):
        return "参数错误: 需要 JSON 对象"

    action = data.get("action", "list")

    # 使用全局变量保存任务列表（在 agent 会话期间有效）
    if not hasattr(task_tool, "_tasks"):
        task_tool._tasks = []  # type: ignore[attr-defined]

    if action == "list":
        if not task_tool._tasks:
            return "任务列表为空。使用 task add 添加任务。"
        lines = []
        for i, t in enumerate(task_tool._tasks, 1):
            status = "✓" if t.get("done") else "⬜"
            lines.append(f"  {status} {i}. {t['content']}")
        done = sum(1 for t in task_tool._tasks if t.get("done"))
        lines.append(f"\n{done}/{len(task_tool._tasks)} 已完成")
        return "\n".join(lines)

    elif action == "add":
        content = data.get("content", "")
        if not content:
            return "错误: add 需要 content 字段"
        task_tool._tasks.append({"content": content, "done": False})
        return f"已添加任务 #{len(task_tool._tasks)}: {content}"

    elif action == "done":
        idx = data.get("index", 0)
        if not isinstance(idx, int) or idx < 1 or idx > len(task_tool._tasks):
            return f"错误: index 必须在 1-{len(task_tool._tasks)} 之间"
        task_tool._tasks[idx - 1]["done"] = True
        return f"已完成任务 #{idx}: {task_tool._tasks[idx - 1]['content']}"

    else:
        return f"未知操作: {action}。支持: list, add, done"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 工具 JSON Schema（OpenAI Function Calling 格式）
# ═══════════════════════════════════════════════════════════════════════════════
# 每个工具需要定义 name / description / parameters (JSON Schema)。
# 函数签名和工具实现之间的桥接由 _dispatch_tool_call 完成。
# 工具实现函数保持原样不变。

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "在文件中搜索匹配的文本模式，返回 文件路径:行号:内容。"
                "path 可以是文件或目录，include 用于过滤文件类型（如 **/*.py 递归匹配）。"
                "结果超过上限时会列出未搜索的文件，方便缩小范围。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "搜索的正则表达式或字面文本，如 'def calculate' 或 'TODO'",
                    },
                    "path": {
                        "type": "string",
                        "description": "文件或目录路径，默认为当前目录。可以是具体文件。",
                    },
                    "include": {
                        "type": "string",
                        "description": "文件匹配 glob，如 '**/*.py' 递归搜索所有 Python 文件",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": (
                "按 glob 模式查找文件列表。自动跳过隐藏目录和 node_modules/.venv 等。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "glob 匹配模式，如 '**/*.py' 或 '*.json'",
                    },
                    "path": {
                        "type": "string",
                        "description": "搜索的起始目录，默认为当前目录",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": (
                f"读取文件内容。传文件路径返回内容（>{READ_MAX_BYTES} 字节自动截断），"
                "传目录路径则列出目录中的文件列表。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对于 workspace 的文件或目录路径",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": (
                "创建或覆盖文件。会自动创建父目录。注意: 这会覆盖已有文件！"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对于 workspace 的文件路径",
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的完整文件内容",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": (
                "在文件中进行精确字符串替换。old_string 必须唯一匹配（除非设置 replace_all: true）。"
                "示例: 将 'def old_name():' 替换为 'def new_name():'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要编辑的文件路径",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "要被替换的原文本，必须和文件中的内容完全一致（含缩进）",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "替换后的新文本",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "是否替换所有匹配项，默认 false（要求唯一匹配）",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute",
            "description": (
                f"在 workspace 下执行 shell 命令。{EXECUTE_TIMEOUT} 秒超时。"
                "禁止执行 rm -rf、git push --force、sudo 等危险命令。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "要执行的 shell 命令",
                    },
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": (
                "管理任务清单（Todo 列表）用于跟踪进度。"
                "action: list(查看) / add(添加,需 content) / done(完成,需 index)"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "add", "done"],
                        "description": "操作类型",
                    },
                    "content": {
                        "type": "string",
                        "description": "任务描述（add 时必需）",
                    },
                    "index": {
                        "type": "integer",
                        "description": "任务序号（done 时必需）",
                    },
                },
                "required": ["action"],
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. System Prompt
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""\
你是一个软件工程助手，类似 Claude Code。你可以使用工具来探索文件、搜索代码、修改文件和执行命令。

## 工作原则

1. **先探索再行动** — 在修改前先 read/grep/glob 了解代码结构
2. **精确修改** — 使用 edit 做精确替换，使用 write 创建新文件
3. **任务拆分** — 用 task 工具管理复杂任务的进度
4. **验证结果** — 修改后用 execute 运行测试/lint 确认变更正确
5. **安全第一** — 不执行破坏性命令，不修改 workspace 外的文件

## 高效使用工具的建议

1. **grep 是主力搜索工具** — 搜索代码内容首选 grep，不要逐个 read 文件来找东西
2. **glob 发现文件，grep 搜索内容** — 先用 glob 了解文件结构，再用 grep 精准定位
3. **grep 的 path 可以是文件** — 如果知道具体文件，直接传文件路径，比传目录+include 更高效
4. **include 用 **/*.ext 递归匹配** — `"include": "**/*.py"` 会递归所有子目录
5. **结果被截断时缩小范围** — 看截断信息里列出的未搜索文件，对它们分目录重新搜索
6. **read 用于阅读完整文件** — 当你需要理解文件整体结构时用 read，而不是用它来搜索

## 规则

- 先用工具收集信息，再做出回答
- 如果工具失败，分析原因并尝试其他方法
- 始终在 workspace 内操作，不要访问外部路径
- 完成任务后总结你做了什么""")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Function Calling 引擎
# ═══════════════════════════════════════════════════════════════════════════════

def _dispatch_tool_call(tool_name: str, arguments: str) -> str:
    """将 OpenAI 的 tool_calls.function.arguments (JSON 字符串) 桥接到工具函数。

    工具函数分为两类:
      - 接受 JSON 字符串的 (grep, glob, write, edit, task): 直接把 arguments 原样传入
      - 接受纯字符串的 (read, execute): 从 arguments JSON 中提取对应字段
    """
    if tool_name not in _tools:
        available = ", ".join(_tools.keys())
        return f"未知工具「{tool_name}」。可用: {available}"

    try:
        # arguments 是 JSON 字符串，如 '{"pattern": "def ", "include": "**/*.py"}'
        args_dict = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        # 如果解析失败，当作纯字符串传入
        return _tools[tool_name]["fn"](arguments)

    # 对于单参数工具，提取第一个参数值直接传入
    single_param_tools = {
        "read": "path",
        "execute": "cmd",
    }

    if tool_name in single_param_tools:
        key = single_param_tools[tool_name]
        value = args_dict.get(key, "")
        return _tools[tool_name]["fn"](value)

    # 对于 JSON 参数工具，把整个 args_dict 转成 JSON 字符串传入
    try:
        return _tools[tool_name]["fn"](json.dumps(args_dict, ensure_ascii=False))
    except Exception as e:
        return f"执行「{tool_name}」时出错: {e}"


def _save_log(messages: list[dict], tag: str) -> Path:
    log_dir = Path(__file__).parent / "conversation_log"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"agent_fc_{tag}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"tag": tag, "timestamp": ts, "messages": messages},
                   f, ensure_ascii=False, indent=2)
    return path


def agent_run(task: str, model: str | None = None, workspace: str | None = None) -> str | None:
    """使用 OpenAI Function Calling 运行 agent。

    Args:
        task: 用户的任务描述。
        model: 模型名称，默认从环境变量读取。需要支持 function calling 的模型。
        workspace: 工作目录，默认为当前目录。

    Returns:
        最终答案字符串，或 None（循环耗尽）。
    """
    if workspace:
        _set_workspace(Path(workspace))
    else:
        _set_workspace(Path.cwd())

    model = model or os.getenv("MODEL") or "gpt-4o-mini"
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Workspace: {_workspace}\n\n"
            f"任务: {task}\n\n"
            f"请开始执行。先探索必要的文件，然后用工具完成任务。"
        )},
    ]

    # 对话日志（用于事后审查）
    conversation_log: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    print(f"\n{'=' * 60}")
    print(f"  Agent — Function Calling 模式")
    print(f"{'=' * 60}")
    print(f"  Workspace: {_workspace}")
    print(f"  任务: {task[:80]}{'...' if len(task) > 80 else ''}")
    print(f"  可用工具: {', '.join(_tools.keys())}")
    print(f"{'=' * 60}\n")

    tool_call_count = 0

    for step in range(1, MAX_STEPS + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            temperature=0.2,
        )

        msg = response.choices[0].message

        # 打印模型文本输出（思考过程）
        if msg.content:
            content_preview = msg.content[:200]
            print(f"─── 第 {step} 步 ───")
            print(f"  💭 {content_preview}")
            if len(msg.content) > 200:
                print(f"  ... (共 {len(msg.content)} 字符)")

        # 如果没有 tool_calls → 最终答案
        if not msg.tool_calls:
            print(f"\n{'=' * 60}")
            print(f"  ✅ 完成 ({step} 步, {tool_call_count} 次工具调用)")
            print(f"{'=' * 60}")
            if msg.content:
                print(f"\n{msg.content}\n")
            conversation_log.append({"role": "assistant", "content": msg.content})
            _save_log(conversation_log, "completed")
            return msg.content

        # 处理 tool_calls
        print(f"─── 第 {step} 步 ───")
        if msg.content:
            print(f"  💭 {msg.content[:120]}")

        # 将 assistant 消息（含 tool_calls）加入对话
        messages.append(msg)
        log_entry = {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        }
        conversation_log.append(log_entry)

        # 执行每个工具调用
        for tc in msg.tool_calls:
            tool_call_count += 1
            tool_name = tc.function.name
            arguments = tc.function.arguments

            args_preview = arguments[:100]
            print(f"  🔧 {tool_name}: {args_preview}")

            observation = _dispatch_tool_call(tool_name, arguments)

            obs_len = len(observation)
            obs_preview = observation[:400].replace("\n", "\n  ")
            print(f"  📋 结果 ({obs_len} 字符):")
            print(f"  {obs_preview}")
            if obs_len > 400:
                print(f"  ... (已截断显示, 共 {obs_len} 字符)")
            print()

            # 将工具结果加入对话
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": observation,
            })
            conversation_log.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": observation,
            })

    print(f"\n已达到最大步数 ({MAX_STEPS})，未能完成任务。")
    _save_log(conversation_log, "max_steps")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CLI 入口
# ═══════════════════════════════════════════════════════════════════════════════

DEMO_TASKS = [
    ("🔍 代码搜索", "帮我找出这个项目里所有的 Python 函数定义，按文件分组列出"),
    ("📂 文件探索", "看看 data 目录里有什么数据文件，读一下并告诉我内容概要"),
    ("🔧 代码修改", "创建一个 hello.py 文件，包含一个 greet(name) 函数，然后运行它"),
    ("📊 项目分析", "分析这个项目的结构和代码量（文件数、行数等）"),
    ("✏️  自定义任务", None),
]

if __name__ == "__main__":
    import sys

    print()
    print("  Claude Code 风格 Agent (Function Calling)")
    print("  ===========================================")
    print(f"  工具: grep, glob, read, write, edit, execute, task")
    print(f"  模式: OpenAI Function Calling（结构化 tool_calls）")
    print(f"  最大步数: {MAX_STEPS}")
    print()

    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        print("选择演示任务:\n")
        for i, (tag, q) in enumerate(DEMO_TASKS, 1):
            print(f"  {i}. {tag} — {q or '(自定义)'}")
        print()
        try:
            choice = int(input("请输入编号 [1-5]: ").strip())
            if 1 <= choice < len(DEMO_TASKS):
                task = DEMO_TASKS[choice - 1][1]
            else:
                task = input("请输入任务: ").strip()
        except (ValueError, IndexError):
            task = input("请输入任务: ").strip()

    if not task:
        print("任务为空，退出。")
        sys.exit(0)

    try:
        agent_run(task)
    except RuntimeError as e:
        print(f"\n错误: {e}")
        print("请在 .env 文件中设置 OPENAI_API_KEY")
    except KeyboardInterrupt:
        print("\n\n用户中断。")
