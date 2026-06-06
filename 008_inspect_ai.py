"""
使用 Inspect AI 评估 Agent 工具调用能力的 demo。

Inspect AI 是 UK AISI 开源的 LLM 评估框架，核心理念：
  Dataset（数据集） → Solver（求解器/Agent） → Scorer（评分器）

这个 demo 展示最核心的用法：
  1. 用 @tool 装饰器定义一个计算器工具（带错误注入，模拟真实 API）
  2. 用 basic_agent 作为 solver（Inspect 内置的 ReAct agent loop）
  3. 用 math.isclose 容差比较评分（解决浮点精度不一致问题）
  4. 单模型评测，结果输出到 results/ 目录

运行方式：
  uv run 008_inspect_ai.py              # 默认 10 题，0% 错误率
  uv run 008_inspect_ai.py --limit 5    # 5 题
  uv run 008_inspect_ai.py --error-rate 0.2  # 20% 工具失败率，测 error recovery

查看结果：
  uv run inspect view --log-dir results/latest
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"

from dotenv import load_dotenv
load_dotenv()

# 简单的数学题数据集（题目 → 答案）
MATH_PROBLEMS = [
    ("15 * 8 + 12", "132"),
    ("83.5 * 12 + 17 - 9.8", "1009.2"),
    ("一个半径为 5 的圆的面积是多少？用 pi * 5**2 计算", "78.53981633974483"),
    ("计算 (3**2 + 4**2) 的平方根", "5.0"),
    ("100 / 3 + 7 * 8", "89.33333333333333"),
    ("2**10 - 1", "1023"),
    ("(100 - 25) * 0.8 + 20", "80.0"),
    ("99 * 99", "9801"),
    ("1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10", "55"),
    ("365 * 24", "8760"),
    ("sqrt(144) + 10", "22.0"),
    ("3.14159 * 10**2", "314.159"),
    ("(50 + 50) * 0 / 999 + 42", "42.0"),
    ("7 * 8 * 9 * 10 / 5040", "1.0"),
    ("1024 / 2 / 2 / 2 / 2 / 2", "32.0"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 第一节：用 @tool 装饰器定义工具
# ═══════════════════════════════════════════════════════════════════════════════
#
# Inspect AI 的 @tool 装饰器自动从函数签名和 docstring 生成 tool schema。
# 参考: https://inspect.ai-safety-institute.org.uk/tools/

def _build_task(num_problems: int, error_rate: float = 0.2):
    """构建 Inspect Task。延迟 import 避免模块顶层依赖。"""
    import math
    import re

    from inspect_ai import Task, task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.scorer import Scorer, Target, scorer, mean, stderr
    from inspect_ai.solver import TaskState, basic_agent
    from inspect_ai.tool import tool

    # 模拟现实 API 的 5 种典型失败模式
    FAKE_ERRORS = [
        "Error: numeric overflow, expression too large",
        "Error: invalid expression syntax, please rephrase",
        "Error: timeout, please retry",
        "Error: division by zero",
        "Error: rate limited, wait and retry",
    ]

    @tool
    def calculator():
        """计算器工具。20% 概率失败时请重试或改写表达式。"""

        async def execute(expression: str) -> str:
            """
            执行算术表达式。
            Args:
              expression: 形如 "3 * 8 + 2" 的算术字符串
            """
            if error_rate > 0 and random.random() < error_rate:
                return random.choice(FAKE_ERRORS)

            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return "Error: 表达式只能含数字和 + - * / . ( )"
            try:
                result = eval(expression, {"__builtins__": {}}, {})
                return str(result)
            except Exception as e:
                return f"Error: {e}"

        return execute

    # 构建数据集
    samples = []
    for i, (question, answer) in enumerate(MATH_PROBLEMS[:num_problems]):
        samples.append(Sample(
            id=f"math_{i:02d}",
            input=question,
            target=answer,
        ))

    # ── 自定义评分器：带容差的数值比较 ──
    @scorer(metrics=[mean(), stderr()])
    def numeric_match() -> Scorer:
        """提取输出中的数值，用 math.isclose 做容差比较。

        解决浮点精度不一致问题：
          89.33333333333333 vs 89.33333333333334 → 判为正确
          80.0 vs 80                             → 判为正确
        """

        async def score(state: TaskState, target: Target) -> dict:
            output = state.output.completion or ""
            expected = target.text

            # 从字符串中提取所有数值
            output_nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', output)
            target_nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', expected)

            if not output_nums or not target_nums:
                return {"value": "I", "explanation": "无法提取数值"}

            # 取最后一个数值作为答案（模型通常在末尾给出答案）
            answer = float(output_nums[-1])
            gold = float(target_nums[0])

            if math.isclose(answer, gold, rel_tol=1e-9, abs_tol=1e-9):
                return {"value": "C", "explanation": f"匹配: {answer} ≈ {gold}"}
            else:
                return {"value": "I",
                        "explanation": f"不匹配: {answer} ≠ {gold} (Δ={abs(answer-gold):.2e})"}

        return score

    @task
    def math_agent_task():
        return Task(
            dataset=MemoryDataset(samples),
            solver=basic_agent(
                tools=[calculator()],
                max_attempts=1,
                message_limit=20,
            ),
            scorer=numeric_match(),
        )

    return math_agent_task()


# ═══════════════════════════════════════════════════════════════════════════════
# 第二节：运行评测
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect AI Agent 工具调用评测")
    parser.add_argument("--limit", type=int, default=10, help="题目数量（默认 10）")
    parser.add_argument("--error-rate", type=float, default=0.0,
                        help="工具错误注入概率（0~1，默认 0）")
    parser.add_argument("--model", type=str, default=None,
                        help="模型名（默认从 .env 的 MODEL 读取）")
    args = parser.parse_args()

    # 从 .env 读取配置
    model = args.model or os.getenv("MODEL") or "gpt-4o-mini"
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("请在 .env 中设置 OPENAI_API_KEY")
        return 1

    os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url

    # 准备输出目录
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    work_dir = RESULTS_DIR / ts
    work_dir.mkdir(parents=True, exist_ok=True)
    latest = RESULTS_DIR / "latest"
    latest.unlink(missing_ok=True)
    latest.symlink_to(ts)

    print(f"\n=== Inspect AI Math Agent === 模型: {model}  题数: {args.limit}")
    print(f"  错误注入率: {args.error_rate:.0%}  结果目录: {work_dir}\n")

    from inspect_ai import eval

    task = _build_task(args.limit, error_rate=args.error_rate)
    logs = eval(
        task,
        model=f"openai/{model}",
        log_dir=str(work_dir),
        max_connections=10,
        max_samples=10,
        display="plain",
    )

    # 输出结果
    print("\n" + "=" * 60)
    print("评测结果")
    print("=" * 60)
    for log in logs:
        if log.results:
            score = log.results.scores[0].metrics.get("accuracy")
            acc = score.value if score else None
            samples = log.results.total_samples
            completed = log.results.completed_samples
            if acc is not None:
                print(f"  正确率: {acc:.0%} ({completed}/{samples})")
        else:
            print("  ❌ 无结果")

    print(f"\n详细报告：uv run inspect view --log-dir {work_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
