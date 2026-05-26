"""
无限对话流 —— ReAct Agent 的上下文压缩与长对话管理。

核心机制：
  - Token 预算检查：每轮 reasoning 前检查 token 用量
  - LLM 压缩（pre_reasoning）：超阈值时用 LLM 生成压缩摘要
  - 消息持久化 + 内存释放：被压缩的消息落盘后从活跃上下文移除

核心概念：
  - compressed_summary 生命周期：生成 → 注入 → 更新 → 重置
  - 压缩摘要以 user 消息形式注入 LLM 上下文
"""

from __future__ import annotations

import datetime
import json
import os
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# 0. 常量与配置
# ═══════════════════════════════════════════════════════════════════════════════

MAX_INPUT_LENGTH = 4096          # 模型最大输入 token 数（模拟）
COMPACT_THRESHOLD_RATIO = 0.8     # 触发压缩的 token 占用比例
RESERVE_THRESHOLD_RATIO = 0.15    # 压缩后保留的 token 比例


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 对话内存
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConversationMemory:
    """对话历史存储。

    messages 存放活跃消息（发送给 LLM 的上下文）。
    compressed_summary 存放 LLM 生成的压缩摘要。
    当消息被压缩后，从 messages 中移除，信息浓缩进 compressed_summary。
    """
    messages: list[dict] = field(default_factory=list)
    compressed_summary: str = ""

    def add(self, msg: dict) -> None:
        self.messages.append(msg)

    def get_messages(self) -> list[dict]:
        return self.messages

    def compact(self, keep_count: int) -> list[dict]:
        """保留最后 keep_count 条消息，返回被移除的消息（用于落盘）。"""
        if keep_count >= len(self.messages):
            return []
        removed = self.messages[:-keep_count]
        self.messages = self.messages[-keep_count:]
        return removed

    def update_compressed_summary(self, summary: str) -> None:
        self.compressed_summary = summary

    def clear_all(self) -> None:
        self.messages.clear()
        self.compressed_summary = ""


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Token 估算器
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    """简易 token 估算：中文约 1.5 字/token，英文约 4 字/token。"""
    chinese_chars = sum(1 for c in text if "一" <= c <= "鿿")
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """估算消息列表的总 token 数。"""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    total += estimate_tokens(block["text"])
    return total


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 上下文压缩引擎
# ═══════════════════════════════════════════════════════════════════════════════

class ContextCompressor:
    """LLM 驱动的上下文压缩。

    参考 QwenPaw 的 LightContextManager.pre_reasoning():
      1. 计算 token 占用量
      2. 如果超过 compact_threshold → 拆分消息
      3. 调用 LLM 生成结构化压缩摘要
      4. 更新 compressed_summary，从内存中移除旧消息

    压缩摘要格式（由专用 Compactor prompt 生成）:
      ## 目标
      ## 进展（已完成 / 进行中）
      ## 关键发现与决策
      ## 下一步
      ## 关键上下文（文件路径、变量名等）
    """

    COMPACTOR_SYSTEM_PROMPT = textwrap.dedent("""\
    你是一个对话压缩器。你的任务是将一段冗长的对话历史压缩为结构化的摘要。

    压缩摘要必须包含以下部分：

    ## 目标
    [用户的核心任务是什么]

    ## 进展
    ### 已完成
    - [已完成的事项]

    ### 进行中
    - [当前正在做的事]

    ## 关键发现与决策
    - [重要发现、决策及其理由]

    ## 下一步
    1. [接下来要做什么]

    ## 关键上下文
    - [继续工作需要的文件路径、变量名、数据等]

    规则：
    - 保留关键信息，丢弃冗余的中间推理过程
    - 只输出上述格式的摘要，不要输出其他内容
    - 如果已有之前的摘要，在其基础上更新而非重写
    """)

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def compact(
        self,
        messages_to_compact: list[dict],
        previous_summary: str,
    ) -> str:
        """调用 LLM 将待压缩消息生成结构化摘要。"""
        prev = f"\n\n## 之前的摘要\n{previous_summary}" if previous_summary else ""
        conversation_text = self._messages_to_text(messages_to_compact)

        user_prompt = (
            f"请压缩以下对话历史。{prev}\n\n"
            f"## 对话历史\n\n{conversation_text}\n\n"
            f"请生成结构化压缩摘要："
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.COMPACTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"  [压缩失败: {e}]")
            return ""

    @staticmethod
    def _messages_to_text(messages: list[dict]) -> str:
        lines: list[str] = []
        for msg in messages:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            # 截断过长的单条消息
            if len(content) > 2000:
                content = content[:2000] + "..."
            lines.append(f"[{role}]: {content}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 对话持久化
# ═══════════════════════════════════════════════════════════════════════════════

def _save_conversation(messages: list[dict], tag: str = "compressed") -> None:
    """将压缩后的消息保存到 conversation_log 目录。"""
    log_dir = Path(__file__).parent / "conversation_log"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = log_dir / f"infinite_{tag}_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({"messages": messages}, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 无限流对话 Agent
# ═══════════════════════════════════════════════════════════════════════════════

class InfiniteConversation:
    """带上下文压缩的对话 Agent —— 支持无限长对话。

    核心机制：
      - 每轮对话前检查 token 预算，超阈值时触发 LLM 压缩
      - 旧消息浓缩为结构化摘要，以 user 消息形式注入上下文
      - 被压缩的消息落盘到 conversation_log/，从活跃上下文移除
    """

    SYSTEM_PROMPT = textwrap.dedent("""\
    你是一个友好的 AI 助手。请认真回答用户的问题。

    如果上下文中包含 [压缩摘要]，它记录了之前对话的关键信息，你可以参考它来保持对话的连贯性。
    """)

    def __init__(self, model: str | None = None):
        model = model or os.getenv("MODEL") or "gpt-4o-mini"
        self.model = model
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

        self.memory = ConversationMemory()
        self.compressor = ContextCompressor(self.client, model)
        self._compact_count = 0

    # ── Token 预算管理 ──────────────────────────────────────────────────────

    def _calculate_token_usage(self) -> dict:
        """计算当前上下文的 token 占用详情。"""
        sys_tokens = estimate_tokens(self.SYSTEM_PROMPT)
        summary_tokens = estimate_tokens(self.memory.compressed_summary)
        msg_tokens = estimate_messages_tokens(self.memory.get_messages())
        return {
            "system_prompt": sys_tokens,
            "compressed_summary": summary_tokens,
            "messages": msg_tokens,
            "total": sys_tokens + summary_tokens + msg_tokens,
        }

    # ── pre_reasoning: Token 预算检查 + 压缩 ─────────────────────────────────

    def _check_and_compact(self) -> bool:
        """检查 token 用量，超阈值时触发 LLM 压缩。返回 True 表示发生了压缩。"""
        usage = self._calculate_token_usage()
        threshold = int(MAX_INPUT_LENGTH * COMPACT_THRESHOLD_RATIO)

        print(f"  [Token: {usage['total']}/{MAX_INPUT_LENGTH} "
              f"(阈值: {threshold}), "
              f"sys={usage['system_prompt']}, "
              f"summary={usage['compressed_summary']}, "
              f"msgs={usage['messages']}]")

        if usage["total"] <= threshold:
            return False

        print(f"  ⚠️  Token 超阈值，触发上下文压缩...")
        return self._do_compact()

    def _do_compact(self) -> bool:
        """执行上下文压缩。"""
        active = self.memory.get_messages()
        reserve = int(MAX_INPUT_LENGTH * RESERVE_THRESHOLD_RATIO)

        # 从后向前遍历，找出保留区
        keep_idx = len(active)
        keep_tokens = 0
        for i in range(len(active) - 1, -1, -1):
            msg_tokens = estimate_tokens(active[i].get("content", "") or "")
            if keep_tokens + msg_tokens > reserve:
                break
            keep_tokens += msg_tokens
            keep_idx = i

        if keep_idx == 0:
            print("  [跳过压缩: 无可压缩的消息]")
            return False

        messages_to_compact = active[:keep_idx]
        messages_to_keep = active[keep_idx:]

        print(f"  待压缩: {len(messages_to_compact)} 条, 保留: {len(messages_to_keep)} 条")

        # 调用 LLM 生成压缩摘要
        new_summary = self.compressor.compact(
            messages_to_compact, self.memory.compressed_summary
        )

        if not new_summary:
            print("  [Fallback: 压缩返回空，跳过本次压缩]")
            return False

        # 持久化旧消息
        _save_conversation(messages_to_compact)

        # 压缩：移除旧消息，只保留最近的消息
        self.memory.compact(len(messages_to_keep))

        # 更新压缩摘要
        self.memory.update_compressed_summary(new_summary)
        self._compact_count += 1

        print(f"  ✅ 压缩完成 (#{self._compact_count}), "
              f"摘要长度: {len(new_summary)} 字符")
        return True

    # ── 消息格式化 ──────────────────────────────────────────────────────────

    def _format_messages_for_llm(self) -> list[dict]:
        """构建发给 LLM 的消息列表。

        关键设计（对应 QwenPaw 的 get_memory(prepend_summary=True)）：
          [system_prompt] + [compressed_summary 作为 user 消息] + [活跃消息]

        压缩摘要以 user 消息形式注入，LLM 可以自然理解。
        """
        messages: list[dict] = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ]

        if self.memory.compressed_summary:
            messages.append({
                "role": "user",
                "content": f"[压缩摘要 - 以下是之前对话的关键信息]\n\n{self.memory.compressed_summary}",
            })

        messages.extend(self.memory.get_messages())
        return messages

    # ── 对话入口 ────────────────────────────────────────────────────────────

    def chat(self, user_input: str) -> str:
        """处理用户输入，返回 AI 回复。"""
        self.memory.add({"role": "user", "content": user_input})

        print(f"\n{'=' * 60}")
        print(f"问：{user_input}")
        print(f"{'=' * 60}\n")

        # 检查是否需要压缩
        self._check_and_compact()

        # 构建消息并调用 LLM
        messages = self._format_messages_for_llm()
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.7,
        )
        reply = response.choices[0].message.content or ""

        print(f"答：{reply}")

        self.memory.add({"role": "assistant", "content": reply})
        return reply

    def save_full_conversation(self) -> None:
        """保存完整对话记录（含压缩摘要和活跃消息）到 conversation_log/。"""
        log_dir = Path(__file__).parent / "conversation_log"
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = log_dir / f"full_conversation_{timestamp}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({
                "compressed_summary": self.memory.compressed_summary,
                "compact_count": self._compact_count,
                "active_messages": self.memory.get_messages(),
            }, f, ensure_ascii=False, indent=2)
        print(f"对话记录已保存至: {filename}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 交互式入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    agent = InfiniteConversation()

    print("=" * 60)
    print("  无限对话流 Demo —— 上下文压缩")
    print("=" * 60)
    print()
    print("核心机制:")
    print("  - Token 预算检查 + LLM 压缩（80% 阈值触发）")
    print("  - 压缩摘要注入为 user 消息")
    print("  - 压缩消息落盘: conversation_log/infinite_compressed_*.json")
    print("  - 退出时保存完整对话: conversation_log/full_conversation_*.json")
    print()
    print("开始对话吧，按 Ctrl+C 退出。")
    print()

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            agent.save_full_conversation()
            break

        if not user_input:
            continue

        reply = agent.chat(user_input)
        print(f"\n🤖 {reply}\n")


if __name__ == "__main__":
    main()
