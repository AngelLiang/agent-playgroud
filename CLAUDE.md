# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个 ReAct（Reasoning + Acting）Agent 模式的教学演示项目，展示 LLM 如何通过"思考-行动-观察"循环来调用工具并回答问题。

## 技术栈

- Python ≥3.11，使用 `uv` 管理依赖
- 依赖：`openai`、`python-dotenv`

## 常用命令

```bash
uv run react.py              # 运行交互式菜单，选择演示问题
uv run react.py "你的问题"    # 直接向 Agent 提问
uv add <包名>                 # 添加新依赖
```

## 架构

[react.py](react.py) 是整个项目的核心，包含三个层次：

1. **工具注册表**（装饰器模式） — `@tool(name, description)` 装饰器将函数注册到全局 `_tools` 字典，引擎自动发现所有已注册工具
2. **ReAct 引擎** — `react()` 函数实现核心循环：
   - 构建 system prompt（包含工具描述列表）
   - 调用 LLM 获取 Thought/Action/Action Input
   - 执行工具，将 Observation 反馈给模型
   - 循环直到出现 Final Answer 或达到 `MAX_STEPS`（10 步）
3. **内置工具** — 三个示例工具：
   - `calculate`：通过 AST 白名单安全地计算数学表达式（仅允许特定节点类型，禁止任意代码执行）
   - `get_current_time`：返回当前日期时间
   - `search_facts`：硬编码知识库的模糊匹配查询

## 关键设计决策

- **安全性**：`calculate` 工具使用 AST 白名单机制，只允许安全节点（算术运算 + `math.*` 属性），拒绝所有其他语法结构
- **输出格式**：LLM 按固定格式输出（`Thought:`/`Action:`/`Action Input:`/`Final Answer:`），通过正则解析，而非 JSON 或 function calling
- **对话记录**：每次运行自动将完整对话历史保存到 `conversation_log/` 目录，便于调试

## 环境变量

复制 `.env.example` 为 `.env` 并填写：

- `OPENAI_API_KEY`：必需
- `OPENAI_BASE_URL`：可选，用于自定义 API 端点
- `MODEL`：可选，默认 `gpt-4o-mini`
