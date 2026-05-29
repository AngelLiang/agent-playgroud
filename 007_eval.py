"""
使用 LLM-as-Judge 评估 ReAct Agent 的示例。

评估维度（文本模式，适用于所有 OpenAI 兼容 API）：
  1. 工具选择评估 — Agent 是否为任务选择了正确的工具
  2. 忠实度评估 — 回答是否忠实于工具结果，有无幻觉
  3. 正确性评估 — 答案是否事实正确

自定义评估器（纯代码，不依赖 LLM）：
  4. 步骤效率 — Agent 是否用最少步骤完成任务
  5. Token 效率 — Token 消耗评估
  6. 答案完整性 — 自定义 LLM-as-Judge 评估器

每次运行自动保存完整对话历史到 conversation_log/ 目录，方便事后审查
每一轮发给 LLM 的消息（system prompt、工具返回等）。

运行方式：
  uv run python 007_eval.py                              # 运行全部用例
  uv run python 007_eval.py --difficulty hard            # 只跑困难用例
  uv run python 007_eval.py --model gpt-4o               # 指定 Agent 模型
  uv run python 007_eval.py --eval-model gpt-4o-mini     # 指定评估裁判模型
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Phoenix Evals：LLM-as-Judge 评估器 ────────────────────────────────────
from phoenix.evals import (
    LLM,
    LLMEvaluator,
    create_evaluator,
)

# ── 复用 001_react 中的 ReAct 引擎 ──────────────────────────────────────────
import importlib

_react = importlib.import_module("001_react")
_build_system_prompt = _react._build_system_prompt
_execute_tool = _react._execute_tool
_parse_step = _react._parse_step
_tools = _react._tools

# ═══════════════════════════════════════════════════════════════════════════════
# 第一节：可追踪的 ReAct Agent
# ═══════════════════════════════════════════════════════════════════════════════


def traced_react(
    question: str,
    model: str | None = None,
    max_steps: int = 10,
) -> dict[str, Any]:
    """
    运行 ReAct 循环，返回详细的步骤记录用于评估。

    与 001_react.react() 不同，此函数：
      - 返回结构化的步骤记录（而非仅返回最终答案）
      - 记录每次工具调用的输入、输出、延迟
      - 记录模型回复的 token 用量
      - 自动保存完整对话历史到 conversation_log/ 目录

    Returns:
        {
            "question": str,
            "final_answer": str | None,
            "model": str,
            "steps": [...],
            "total_tokens": int,
            "success": bool,
            "messages": [...],       # 完整的 messages 列表（发给 LLM 的每条消息）
            "conversation_file": str, # 保存的对话日志文件路径
        }
    """
    model = model or os.getenv("MODEL") or "gpt-4o-mini"
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    system_prompt = _build_system_prompt()
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    steps: list[dict] = []
    total_tokens = 0
    final_answer = None

    for step_idx in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
        text = response.choices[0].message.content or ""

        # 提取 token 用量
        usage = response.usage
        token_info = (
            {
                "prompt": usage.prompt_tokens,
                "completion": usage.completion_tokens,
                "total": usage.total_tokens,
            }
            if usage
            else {"prompt": 0, "completion": 0, "total": 0}
        )
        total_tokens += token_info["total"]

        parsed = _parse_step(text)

        step_record = {
            "step": step_idx,
            "thought": parsed["thought"],
            "action": parsed["action"],
            "action_input": parsed["action_input"],
            "observation": None,
            "full_response": text,
            "token_usage": token_info,
        }

        # 记录模型回复到 messages
        messages.append({"role": "assistant", "content": text})

        if parsed["final_answer"]:
            final_answer = parsed["final_answer"]
            steps.append(step_record)
            break

        if parsed["action"] and parsed["action_input"] is not None:
            observation = _execute_tool(parsed["action"], parsed["action_input"])
            step_record["observation"] = observation
            obs_msg = {"role": "user", "content": f"Observation: {observation}"}
            messages.append(obs_msg)
        else:
            break

        steps.append(step_record)
    else:
        # 达到 max_steps
        pass

    # ── 保存完整对话历史 ──
    log_dir = Path(__file__).parent / "conversation_log"
    log_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_id = re.sub(r"[^\w\-]", "_", question[:30])
    filename = log_dir / f"eval_{timestamp}_{safe_id}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            {
                "question": question,
                "model": model,
                "success": final_answer is not None,
                "final_answer": final_answer,
                "messages": messages,
                "steps": [
                    {k: v for k, v in s.items() if k != "token_usage"}
                    for s in steps
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "question": question,
        "final_answer": final_answer,
        "model": model,
        "steps": steps,
        "total_tokens": total_tokens,
        "success": final_answer is not None,
        "messages": messages,
        "conversation_file": str(filename),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 第三节：评估数据集
# ═══════════════════════════════════════════════════════════════════════════════


def build_eval_test_cases(difficulty: str = "all") -> list[dict]:
    """
    构建评估用的测试用例。

    每个测试用例包含：
      - question: 用户问题
      - expected_answer: 期望的关键信息（用于正确性评估）
      - expected_tool: 期望调用的工具名（用于工具选择评估）
      - expected_tool_args: 期望的工具参数关键词（用于工具调用评估）
      - difficulty: "easy" | "hard"

    Args:
        difficulty: "easy" 只返回简单用例，"hard" 只返回困难用例，"all" 返回全部
    """
    easy_cases = [
        {
            "id": "case-01-simple-math",
            "question": "计算 15 * 8 + 12 的结果",
            "expected_answer": "132",
            "expected_tool": "calculate",
            "expected_tool_args": "15 * 8 + 12",
            "difficulty": "easy",
        },
        {
            "id": "case-02-knowledge-lookup",
            "question": "Python 的作者是谁？",
            "expected_answer": "Guido van Rossum",
            "expected_tool": "search_facts",
            "expected_tool_args": "python 作者",
            "difficulty": "easy",
        },
        {
            "id": "case-03-compound",
            "question": "一个半径为 5 的圆的面积是多少？",
            "expected_answer": "78.5",
            "expected_tool": "calculate",
            "expected_tool_args": ["pi", "5"],
            "difficulty": "easy",
        },
        {
            "id": "case-04-multi-step",
            "question": "法国首都在什么时区？先查首都，再看时间。",
            "expected_answer": "巴黎",
            "expected_tool": "search_facts",
            "expected_tool_args": "法国首都",
            "difficulty": "easy",
        },
        {
            "id": "case-05-time",
            "question": "现在是什么时间？",
            "expected_answer": None,  # 动态答案，不做精确匹配
            "expected_tool": "get_current_time",
            "expected_tool_args": "",
            "difficulty": "easy",
        },
    ]

    hard_cases = [
        # ── 多步推理：知识库 → 数学计算 ──
        {
            "id": "case-06-knowledge-to-math",
            "question": "生命的意义的数字乘以 3 是多少？",
            "expected_answer": "126",
            "expected_tool": None,  # 需要调用两个工具：search_facts + calculate
            "expected_tool_args": None,
            "difficulty": "hard",
            "min_steps": 2,  # 至少需要2步：先查再算
        },
        {
            "id": "case-07-complex-math",
            "question": "计算 sin(pi/6) + sqrt(25) + log(e**3) 的结果",
            "expected_answer": "8.5",  # 0.5 + 5 + 3
            "expected_tool": "calculate",
            "expected_tool_args": ["sin", "sqrt", "log"],
            "difficulty": "hard",
        },
        # ── 知识库边界：查询不存在的数据 ──
        {
            "id": "case-08-missing-knowledge",
            "question": "Java 的创始人是谁？",
            "expected_answer": None,  # 知识库里没有，Agent 应该诚实告知
            "expected_tool": "search_facts",
            "expected_tool_args": "java",
            "difficulty": "hard",
        },
        # ── 多知识综合：需要查询两个事实并整合 ──
        {
            "id": "case-09-multi-fact",
            "question": "中国首都和法国首都分别是什么？请都查出来。",
            "expected_answer": ["北京", "巴黎"],
            "expected_tool": None,  # 需要调用两次 search_facts
            "expected_tool_args": None,
            "difficulty": "hard",
            "min_steps": 2,
        },
        # ── 数学提取：从工具结果中提取数字再计算 ──
        {
            "id": "case-10-extract-and-math",
            "question": "生命的意义的数字开平方是多少？",
            "expected_answer": "6.48",  # sqrt(42) ≈ 6.48
            "expected_tool": None,  # search_facts → calculate
            "expected_tool_args": None,
            "difficulty": "hard",
            "min_steps": 2,
        },
        # ── 复杂表达式 + 多函数组合 ──
        {
            "id": "case-11-compound-math",
            "question": (
                "计算 floor(sqrt(3**2 + 4**2)) * ceil(pi) 的值，"
                "其中 sqrt 是开平方，floor 是向下取整，ceil 是向上取整"
            ),
            "expected_answer": "20",  # floor(5) * ceil(3.1415...) = 5 * 4 = 20
            "expected_tool": "calculate",
            "expected_tool_args": ["floor", "sqrt", "ceil", "pi"],
            "difficulty": "hard",
        },
    ]

    if difficulty == "easy":
        return easy_cases
    elif difficulty == "hard":
        return hard_cases
    else:
        return easy_cases + hard_cases


# ═══════════════════════════════════════════════════════════════════════════════
# 第四节：构建评估 DataFrame 并运行评估
# ═══════════════════════════════════════════════════════════════════════════════


def build_tool_selection_df(traces: list[dict]) -> pd.DataFrame:
    """
    构建工具选择评估用的 DataFrame。

    ToolSelectionEvaluator 需要的字段：
      - input: 用户输入或对话上下文
      - available_tools: 可用工具列表
      - tool_selection: Agent 选择的工具
    """
    rows = []
    available_tools_desc = "\n".join(
        f"{name}: {info['description']}" for name, info in _tools.items()
    )

    for trace in traces:
        for step in trace["steps"]:
            if step["action"]:
                rows.append(
                    {
                        "input": trace["question"],
                        "available_tools": available_tools_desc,
                        "tool_selection": f"{step['action']}({step['action_input']})",
                        "trace_id": trace["question"],
                    }
                )

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["input", "available_tools", "tool_selection"]
    )


def build_tool_invocation_df(traces: list[dict]) -> pd.DataFrame:
    """
    构建工具调用评估用的 DataFrame。

    ToolInvocationEvaluator 需要的字段：
      - input: 用户输入或对话上下文
      - available_tools: 工具的 schema（JSON 或人类可读格式）
      - tool_selection: Agent 的工具调用
    """
    rows = []
    tool_schemas = {
        "calculate": (
            "calculate: 计算数学表达式。\n"
            "  参数:\n"
            "    - expression (必需): 数学表达式字符串，如 'sqrt(3**2 + 4**2)'"
        ),
        "get_current_time": (
            "get_current_time: 获取当前日期和时间。\n  参数: 无"
        ),
        "search_facts": (
            "search_facts: 按关键词查询知识库。\n"
            "  参数:\n"
            "    - query (必需): 搜索关键词"
        ),
    }
    available_tools_desc = "\n".join(tool_schemas.values())

    for trace in traces:
        for step in trace["steps"]:
            if step["action"]:
                rows.append(
                    {
                        "input": trace["question"],
                        "available_tools": available_tools_desc,
                        "tool_selection": f"{step['action']}({step['action_input']})",
                        "trace_id": trace["question"],
                    }
                )

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["input", "available_tools", "tool_selection"]
    )


def build_tool_response_df(traces: list[dict]) -> pd.DataFrame:
    """
    构建工具响应处理评估用的 DataFrame。

    ToolResponseHandlingEvaluator 需要的字段：
      - input: 用户查询或对话上下文
      - tool_call: Agent 的工具调用
      - tool_result: 工具返回结果
      - output: Agent 的后续处理
    """
    rows = []

    for trace in traces:
        for step in trace["steps"]:
            if step["action"] and step["observation"]:
                # 对于最后一步，output 是最终答案；否则是下一步的 thought
                next_step = next(
                    (
                        s
                        for s in trace["steps"]
                        if s["step"] == step["step"] + 1
                    ),
                    None,
                )
                if next_step:
                    output = next_step["thought"]
                else:
                    output = trace.get("final_answer", "")

                rows.append(
                    {
                        "input": trace["question"],
                        "tool_call": f"{step['action']}({step['action_input']})",
                        "tool_result": step["observation"],
                        "output": output,
                        "trace_id": trace["question"],
                    }
                )

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["input", "tool_call", "tool_result", "output"]
    )


def build_faithfulness_df(traces: list[dict]) -> pd.DataFrame:
    """
    构建忠实度评估用的 DataFrame。

    FaithfulnessEvaluator 需要的字段：
      - input: 用户问题
      - output: Agent 的回答
      - context: 参考上下文（工具返回的所有观测结果）
    """
    rows = []
    for trace in traces:
        if trace["final_answer"]:
            context_parts = []
            for step in trace["steps"]:
                if step["observation"]:
                    context_parts.append(
                        f"[{step['action']}({step['action_input']})]: {step['observation']}"
                    )
            context = "\n".join(context_parts) if context_parts else "无工具调用"

            rows.append(
                {
                    "input": trace["question"],
                    "output": trace["final_answer"],
                    "context": context,
                    "trace_id": trace["question"],
                }
            )
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["input", "output", "context"]
    )


def build_correctness_df(traces: list[dict]) -> pd.DataFrame:
    """
    构建正确性评估用的 DataFrame。

    CorrectnessEvaluator 需要的字段：
      - input: 用户问题
      - output: Agent 回答
    """
    rows = []
    for trace in traces:
        if trace["final_answer"]:
            rows.append(
                {
                    "input": trace["question"],
                    "output": trace["final_answer"],
                    "trace_id": trace["question"],
                }
            )
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["input", "output"]
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 第五节：运行所有评估器
# ═══════════════════════════════════════════════════════════════════════════════


def run_evals(
    traces: list[dict],
    model: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    对所有 trace 运行文本模式 LLM-as-Judge 评估。

    使用 generate_text() + 正则解析，适用于所有 OpenAI 兼容 API，
    不需要 structured output 支持。

    返回字典：key = 评估器名称, value = 评估结果 DataFrame
    """
    model = model or os.getenv("MODEL") or "gpt-4o-mini"
    eval_llm = LLM(
        provider="openai",
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    results: dict[str, pd.DataFrame] = {}
    _run_text_based_evals(traces, eval_llm, results)
    return results


def _run_text_based_evals(
    traces: list[dict],
    eval_llm: LLM,
    results: dict[str, pd.DataFrame],
) -> None:
    """
    使用 LLM.generate_text() 进行基于文本的评估。

    这不需要 structured output 支持，适用于所有 OpenAI 兼容 API。
    缺点是结果解析不如 structured output 可靠。
    """
    import sys
    import time


    # ── 工具选择评估 ──
    tool_selection_df = build_tool_selection_df(traces)
    if not tool_selection_df.empty:
        n = len(tool_selection_df)
        print(f"\n  [工具选择 (文本模式)] 评估 {n} 条 ...")
        scores = []
        for i, (_, row) in enumerate(tool_selection_df.iterrows(), 1):
            # 使用简化的工具列表，避免 prompt 过长
            short_tools = (
                "calculate: 计算数学表达式\n"
                "get_current_time: 获取当前时间\n"
                "search_facts: 查询知识库"
            )
            prompt = (
                "评估 Agent 是否选择了正确的工具。只回复 [correct] 或 [incorrect]。\n"
                f"问题: {row['input']}\n"
                f"可用工具: {short_tools}\n"
                f"选择的工具: {row['tool_selection']}"
            )
            try:
                resp = eval_llm.generate_text(prompt)
                is_correct = resp.strip().lower().startswith("[correct]")
                scores.append({
                    "label": "correct" if is_correct else "incorrect",
                    "score": 1.0 if is_correct else 0.0,
                })
                print(f"    [{i}/{n}] {'✓' if is_correct else '✗'}", end="")
                sys.stdout.flush()
            except Exception as e:
                scores.append({"label": "error", "score": 0.0})
                print(f"    [{i}/{n}] ⚠ {e}", end="")
                sys.stdout.flush()
            time.sleep(0.1)  # 微小延迟避免触发速率限制

        results["工具选择 (文本模式)"] = pd.DataFrame(scores)
        if scores:
            avg = sum(s["score"] for s in scores) / len(scores)
            correct = sum(1 for s in scores if s["label"] == "correct")
            print(f"\n    平均分: {avg:.2f}, 正确: {correct}/{len(scores)}")

    # ── 忠实度评估 ──
    faithfulness_df = build_faithfulness_df(traces)
    if not faithfulness_df.empty:
        n = len(faithfulness_df)
        print(f"\n  [忠实度 (文本模式)] 评估 {n} 条 ...")
        scores = []
        for i, (_, row) in enumerate(faithfulness_df.iterrows(), 1):
            # 截断 context 避免 prompt 过长
            context_short = row["context"][:300]
            prompt = (
                "评估 AI 回答是否基于上下文，没有编造。只回复 [faithful] 或 [unfaithful]。\n"
                f"问题: {row['input']}\n"
                f"上下文: {context_short}\n"
                f"回答: {row['output']}"
            )
            try:
                resp = eval_llm.generate_text(prompt)
                is_faithful = resp.strip().lower().startswith("[faithful]")
                scores.append({
                    "label": "faithful" if is_faithful else "unfaithful",
                    "score": 1.0 if is_faithful else 0.0,
                })
                print(f"    [{i}/{n}] {'✓' if is_faithful else '✗'}", end="")
                sys.stdout.flush()
            except Exception as e:
                scores.append({"label": "error", "score": 0.0})
                print(f"    [{i}/{n}] ⚠ {e}", end="")
                sys.stdout.flush()
            time.sleep(0.5)

        results["忠实度 (文本模式)"] = pd.DataFrame(scores)
        if scores:
            avg = sum(s["score"] for s in scores) / len(scores)
            faithful = sum(1 for s in scores if s["label"] == "faithful")
            print(f"\n    平均分: {avg:.2f}, 忠实: {faithful}/{len(scores)}")

    # ── 正确性评估 ──
    correctness_df = build_correctness_df(traces)
    if not correctness_df.empty:
        n = len(correctness_df)
        print(f"\n  [正确性 (文本模式)] 评估 {n} 条 ...")
        scores = []
        for i, (_, row) in enumerate(correctness_df.iterrows(), 1):
            output_short = row["output"][:200]
            prompt = (
                "评估 AI 回答是否正确。只回复 [correct] 或 [incorrect]。\n"
                f"问题: {row['input']}\n"
                f"回答: {output_short}"
            )
            try:
                resp = eval_llm.generate_text(prompt)
                is_correct = resp.strip().lower().startswith("[correct]")
                scores.append({
                    "label": "correct" if is_correct else "incorrect",
                    "score": 1.0 if is_correct else 0.0,
                })
                print(f"    [{i}/{n}] {'✓' if is_correct else '✗'}", end="")
                sys.stdout.flush()
            except Exception as e:
                scores.append({"label": "error", "score": 0.0})
                print(f"    [{i}/{n}] ⚠ {e}", end="")
                sys.stdout.flush()
            time.sleep(0.5)

        results["正确性 (文本模式)"] = pd.DataFrame(scores)
        if scores:
            avg = sum(s["score"] for s in scores) / len(scores)
            correct = sum(1 for s in scores if s["label"] == "correct")
            print(f"\n    平均分: {avg:.2f}, 正确: {correct}/{len(scores)}")

    return results




# ═══════════════════════════════════════════════════════════════════════════════
# 第六节：自定义评估器
# ═══════════════════════════════════════════════════════════════════════════════


def create_custom_llm_evaluator(eval_llm: LLM) -> LLMEvaluator:
    """
    演示如何用 LLMEvaluator 创建自定义 LLM-as-Judge 评估器。

    与使用内置评估器不同，自定义 LLM 评估器需要你自己定义评估 prompt。
    Phoenix 会用这个 prompt 让 LLM 裁判对你的 Agent 输出打分。
    """
    completness_evaluator = LLMEvaluator(
        name="答案完整性",
        llm=eval_llm,
        prompt_template=(
            "你是一个评估助手。请评估以下回答是否完整地回答了用户的问题。\n"
            "\n"
            "用户问题：{{input}}\n"
            "AI 回答：{{output}}\n"
            "\n"
            "评估标准：\n"
            "- 回答是否直接回应了问题的核心诉求？\n"
            "- 回答是否包含了所有必要的信息？\n"
            "- 回答是否有遗漏或不清不楚的地方？\n"
            "\n"
            "请给出一个分数（0.0 ~ 1.0）和简要说明。\n"
            "1.0 = 完全回答了问题，信息完整\n"
            "0.5 = 部分回答了问题，有遗漏\n"
            "0.0 = 完全没有回答问题\n"
            "\n"
            "分数："
        ),
        schema={"type": "object", "properties": {"score": {"type": "number"}}},
        direction="maximize",
    )
    return completness_evaluator


@create_evaluator(
    name="步骤效率",
    direction="maximize",
    kind="code",
)
def step_efficiency(traces: list[dict]) -> float:
    """评估 Agent 的步骤效率。

    成功用例中，根据步骤数量打分：≤2步=1.0, ≤4步=0.7, ≤6步=0.4, >6步=0.1
    """
    successful = [t for t in traces if t["success"]]
    if not successful:
        return 0.0

    efficiencies = []
    for t in successful:
        steps = len(t["steps"])
        if steps <= 2:
            efficiencies.append(1.0)
        elif steps <= 4:
            efficiencies.append(0.7)
        elif steps <= 6:
            efficiencies.append(0.4)
        else:
            efficiencies.append(0.1)

    return float(np.mean(efficiencies)) if efficiencies else 0.0


@create_evaluator(
    name="Token 效率",
    direction="minimize",
    kind="code",
)
def token_efficiency(traces: list[dict]) -> float:
    """评估 Agent 的 Token 消耗效率（每用例平均 K tokens，越低越好）。"""
    tokens = [t["total_tokens"] for t in traces if t["total_tokens"] > 0]
    return float(np.mean(tokens)) / 1000.0 if tokens else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 第七节：结果可视化与报告
# ═══════════════════════════════════════════════════════════════════════════════


def print_eval_report(
    traces: list[dict],
    eval_results: dict[str, pd.DataFrame],
) -> None:
    """打印完整的评估报告。"""
    print("\n" + "=" * 70)
    print("  ReAct Agent 评估报告")
    print("=" * 70)

    # ── 按难度分组统计 ──
    easy_traces = [t for t in traces if t.get("difficulty") == "easy"]
    hard_traces = [t for t in traces if t.get("difficulty") == "hard"]

    def _trace_stats(trace_list: list[dict], label: str) -> None:
        if not trace_list:
            return
        n = len(trace_list)
        success = sum(1 for t in trace_list if t["success"])
        avg_steps = sum(len(t["steps"]) for t in trace_list) / n
        avg_tokens = sum(t["total_tokens"] for t in trace_list) / n
        max_steps_reached = sum(
            1 for t in trace_list
            if len(t["steps"]) >= 10 and not t["success"]
        )
        print(f"\n  [{label}] {n} 个用例")
        print(f"    成功率: {success}/{n} ({success/n*100:.0f}%)")
        print(f"    平均步骤: {avg_steps:.1f}步, 平均 Token: {avg_tokens:.0f}")
        if max_steps_reached:
            print(f"    达到最大步数: {max_steps_reached} 个")

    print(f"\n  {'─' * 50}")
    print("  按难度统计")
    print(f"  {'─' * 50}")
    _trace_stats(easy_traces, "简单")
    _trace_stats(hard_traces, "困难")

    # ── 困难用例的额外检测 ──
    if hard_traces:
        print(f"\n  {'─' * 50}")
        print("  困难用例多步推理检测")
        print(f"  {'─' * 50}")
        for t in hard_traces:
            required = t.get("min_steps")
            if required is None:
                continue
            actual = len(t["steps"])
            satisfied = "✓" if actual >= required else "✗"
            tools_used = [
                s["action"] for s in t["steps"] if s["action"]
            ]
            print(
                f"  [{satisfied}] {t['question'][:55]}"
            )
            print(f"      需≥{required}步, 实际{actual}步, 工具链: {' → '.join(tools_used)}")

    # ── 每个用例的详情 ──
    print(f"\n{'─' * 70}")
    print("  各用例详情")
    print(f"{'─' * 70}")
    for i, trace in enumerate(traces, 1):
        status = "✓" if trace["success"] else "✗"
        diff = trace.get("difficulty", "?")
        answer = (trace["final_answer"] or "N/A")[:80]
        print(f"  [{status}] [{diff}] 用例{i}: {trace['question'][:50]}")
        print(f"       答案: {answer}")
        print(f"       步骤: {len(trace['steps'])}步, Token: {trace['total_tokens']}")
        for step in trace["steps"]:
            tool_info = (
                f"  → {step['action']}({step['action_input']})"
                if step["action"]
                else "  → Final Answer"
            )
            print(f"        Step {step['step']}: {tool_info}")

    # ── 评估结果汇总 ──
    print(f"\n{'─' * 70}")
    print("  评估指标汇总")
    print(f"{'─' * 70}")

    for eval_name, result_df in eval_results.items():
        if result_df.empty:
            print(f"  [{eval_name}] 无有效数据")
            continue

        score_cols = [c for c in result_df.columns if c.endswith("_score") or c == "score"]
        if not score_cols:
            continue

        scores = result_df[score_cols[0]].dropna()
        if scores.empty:
            continue

        mean_score = scores.mean()
        bar = "█" * int(mean_score * 20) + "░" * (20 - int(mean_score * 20))
        print(f"  [{eval_name}]")
        print(f"    分数: {bar} {mean_score:.2f}")
        print(f"    有效样本: {len(scores)}, 范围: [{scores.min():.2f}, {scores.max():.2f}]")

    print(f"\n{'=' * 70}\n")

    # ── 调优建议 ──
    _print_optimization_suggestions(traces, eval_results)


def _print_optimization_suggestions(
    traces: list[dict],
    eval_results: dict[str, pd.DataFrame],
) -> None:
    """根据评估结果和 trace 数据，自动分析缺陷并给出调优建议。"""
    print("  调优建议（基于评估结果自动分析）")
    print(f"{'─' * 70}")

    suggestions: list[tuple[str, str]] = []

    # 1. 检查是否有达到最大步数的失败用例（死循环/无效重试）
    loop_failures = [
        t for t in traces
        if not t["success"] and len(t["steps"]) >= 10
    ]
    if loop_failures:
        for t in loop_failures:
            tools_used = [s["action"] for s in t["steps"] if s["action"]]
            unique_tools = list(dict.fromkeys(tools_used))  # 去重，保持顺序
            repeated = len(tools_used) > len(unique_tools) * 2
            suggestions.append((
                "🔴 死循环/无效重试",
                f"用例「{t['question'][:40]}」在 {len(t['steps'])} 步后耗尽。\n"
                f"     工具链: {' → '.join(tools_used)}\n"
                f"     → 建议: 对话记录: {t.get('conversation_file', 'N/A')}\n"
                + (
                    f"     → 原因: 工具重复调用，说明 Agent 不理解工具返回结果\n"
                    f"     → 调优: 1) 优化 Observation 的格式，让结果更易解析\n"
                    f"             2) 在 system prompt 中加入示例，教 Agent 如何解读工具输出"
                    if repeated
                    else ""
                )
            ))

    # 2. 检查忠实度低的用例（幻觉）
    faithfulness_key = None
    for key in eval_results:
        if "忠实" in key:
            faithfulness_key = key
            break

    if faithfulness_key is not None:
        df = eval_results[faithfulness_key]
        score_col = [c for c in df.columns if c.endswith("_score") or c == "score"]
        label_col = [c for c in df.columns if c.endswith("_label") or c == "label"]
        if label_col and not df.empty:
            for _, row in df.iterrows():
                if row.get(label_col[0]) == "unfaithful":
                    # 尝试找到对应的 trace（通过 input 匹配）
                    matched = [
                        t for t in traces
                        if t["question"] in str(row.get("input", ""))
                        or str(row.get("input", "")) in t["question"]
                    ]
                    conv_file = matched[0].get("conversation_file", "N/A") if matched else "N/A"
                    suggestions.append((
                        "🟡 幻觉/不忠实",
                        f"回答未忠实于工具结果。\n"
                        f"     → 查看对话记录: {conv_file}\n"
                        f"     → 调优: 1) 在 system prompt 中强调「绝不要编造 Observation」\n"
                        f"             2) 检查工具返回的「未找到」结果是否被 Agent 绕过"
                    ))
                    break  # 只报告一次

    # 3. 检查正确性低的用例
    correctness_key = None
    for key in eval_results:
        if "正确" in key:
            correctness_key = key
            break

    if correctness_key is not None:
        df = eval_results[correctness_key]
        label_col = [c for c in df.columns if c.endswith("_label") or c == "label"]
        if label_col and not df.empty:
            incorrect_count = (df[label_col[0]] != "correct").sum()
            total = len(df)
            if incorrect_count > 0:
                suggestions.append((
                    "🟡 答案错误",
                    f"{incorrect_count}/{total} 个用例被判定为不正确。\n"
                    f"     → 调优: 1) 检查工具能力是否足够（是否需要新增工具）\n"
                    f"             2) 检查 system prompt 的推理指引是否清晰\n"
                    f"             3) 考虑用更强的模型（如 gpt-4o）运行对比测试"
                ))

    # 4. 检查步骤效率（困难用例 vs 简单用例）
    easy_traces = [t for t in traces if t.get("difficulty") == "easy" and t["success"]]
    hard_traces = [t for t in traces if t.get("difficulty") == "hard" and t["success"]]
    if easy_traces and hard_traces:
        easy_avg = sum(len(t["steps"]) for t in easy_traces) / len(easy_traces)
        hard_avg = sum(len(t["steps"]) for t in hard_traces) / len(hard_traces)
        ratio = hard_avg / easy_avg if easy_avg > 0 else 0
        if ratio > 2:
            suggestions.append((
                "🟢 步骤膨胀",
                f"困难用例平均 {hard_avg:.1f} 步，是简单用例 ({easy_avg:.1f} 步) 的 {ratio:.1f} 倍。\n"
                f"     → 调优: 如果多步是合理的（确实需要多个工具），可以接受\n"
                f"             如果是不必要的探索，考虑在 prompt 中加入「先规划再执行」策略"
            ))

    # 5. 总体建议
    if not suggestions:
        print("  ✓ 未发现明显问题，Agent 表现良好！\n")
        return

    for i, (title, detail) in enumerate(suggestions, 1):
        print(f"\n  [{i}] {title}")
        print(f"  {detail}")

    # ── 调试指引 ──
    print(f"\n  {'─' * 50}")
    print("  调试指引")
    print(f"  {'─' * 50}")
    print("  查看对话历史中的具体 prompt 和回复：")
    print("    1. 打开 conversation_log/ 目录")
    print("    2. 找到对应的 eval_*.json 文件")
    print("    3. 检查「messages」字段中的 system prompt 和每轮对话")
    print("    4. 关注「Observation:」消息，确认工具返回格式是否清晰")
    print("")
    print("  调优迭代流程：")
    print("    1. 修改 system prompt → 再次运行评估")
    print("    2. 对比前后两次的评估报告")
    print("    3. 如果某个工具总被误用，检查工具描述是否准确")
    print("    4. 如果模型能力不足，考虑升级模型（--model gpt-4o）")
    print("")



# ═══════════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 LLM-as-Judge 评估 ReAct Agent"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Agent 使用的 LLM 模型（默认从环境变量 MODEL 读取）",
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default=None,
        help="评估裁判模型（默认读取环境变量 MODEL）",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "hard", "all"],
        default="all",
        help="测试用例难度：easy=简单, hard=困难, all=全部（默认）",
    )
    args = parser.parse_args()

    # ── 1. 运行 Agent 获取 traces ──
    test_cases = build_eval_test_cases(difficulty=args.difficulty)
    easy_count = sum(1 for c in test_cases if c.get("difficulty") == "easy")
    hard_count = sum(1 for c in test_cases if c.get("difficulty") == "hard")
    print(f"\n运行 {len(test_cases)} 个测试用例（简单: {easy_count}, 困难: {hard_count}）...")
    print("═" * 50)

    traces: list[dict] = []
    for case in test_cases:
        difficulty_label = f"[{case.get('difficulty', '?')}]"
        print(f"\n▶ {difficulty_label} 用例 {case['id']}: {case['question']}")
        result = traced_react(case["question"], model=args.model)
        result["case_id"] = case["id"]
        result["difficulty"] = case.get("difficulty", "unknown")
        result["min_steps"] = case.get("min_steps", None)
        traces.append(result)
        if result["success"]:
            print(f"  ✓ 答案: {result['final_answer'][:100]}")
            print(f"  ✓ 步骤: {len(result['steps'])}步, Token: {result['total_tokens']}")
        else:
            print(f"  ✗ 未能得出答案")
        print(f"  📝 对话记录: {result['conversation_file']}")

    # ── 2. 运行评估器 ──
    print(f"\n\n{'═' * 50}")
    print("运行 LLM-as-Judge 评估器 ...")
    print("（用 LLM 作为裁判，评估 Agent 回答的质量）")
    print("═" * 50)

    eval_results = run_evals(traces, model=args.eval_model)

    # ── 3. 自定义评估器 ──
    print(f"\n  [自定义评估器]")
    custom_eval_model = args.eval_model or os.getenv("MODEL") or "gpt-4o-mini"

    step_scores = step_efficiency.evaluate({"traces": traces})
    token_scores = token_efficiency.evaluate({"traces": traces})
    print(f"    步骤效率: {step_scores[0].score:.2f}  (分数越高效率越高)")
    print(f"    Token 效率: {token_scores[0].score:.2f} K tokens/用例  (分数越低越好)")

    eval_llm = LLM(
        provider="openai",
        model=custom_eval_model,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    completeness_df = build_correctness_df(traces)
    if not completeness_df.empty:
        n = len(completeness_df)
        print(f"    答案完整性: 评估 {n} 条 ...")
        scores = []
        for i, (_, row) in enumerate(completeness_df.iterrows(), 1):
            output_short = row["output"][:200]
            prompt = (
                "评估 AI 回答是否完整。给出 0.0-1.0 的分数。\n"
                f"问题: {row['input']}\n"
                f"回答: {output_short}\n"
                "格式: [分数] 例如 [0.8]"
            )
            try:
                resp = eval_llm.generate_text(prompt)
                match = re.search(r"\[([\d.]+)\]", resp)
                score_val = float(match.group(1)) if match else 0.5
                scores.append({"score": score_val})
                print(f"    [{i}/{n}] {score_val:.1f}", end="")
                sys.stdout.flush()
            except Exception as e:
                scores.append({"score": 0.0})
                print(f"    [{i}/{n}] ⚠", end="")
                sys.stdout.flush()
            time.sleep(0.5)

        if scores:
            avg = sum(s["score"] for s in scores) / len(scores)
            print(f"\n    答案完整性 平均分: {avg:.2f}")
            eval_results["答案完整性 (文本模式)"] = pd.DataFrame(scores)

    # ── 4. 打印报告 ──
    print_eval_report(traces, eval_results)


if __name__ == "__main__":
    main()
