# This code is Apache 2 licensed:
# https://www.apache.org/licenses/LICENSE-2.0
# https://til.simonwillison.net/llms/python-react-pattern
import os
import re
import httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL") or None,
)

class ChatBot:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        model = os.getenv("MODEL") or "gpt-4o-mini"
        completion = client.chat.completions.create(model=model, messages=self.messages)
        # Uncomment this to print out token usage each time, e.g.
        # {"completion_tokens": 86, "prompt_tokens": 26, "total_tokens": 112}
        # print(completion.usage)
        return completion.choices[0].message.content

prompt = """
你运行在一个 Thought（思考）、Action（行动）、PAUSE（暂停）、Observation（观察）的循环中。
在循环结束时，你需要输出一个 Answer（答案）。
使用 Thought 来描述你对所问问题的思考过程。
使用 Action 来执行一个可用的行动 - 然后返回 PAUSE。
Observation 将是执行这些行动后的结果。

你可用的行动有：

calculate:
例如: calculate: 4 * 7 / 3
执行一个数学计算并返回数字 - 使用 Python 语法，必要时使用浮点数

wikipedia:
例如: wikipedia: 北京
从维基百科搜索并返回摘要

simon_blog_search:
例如: simon_blog_search: Django
在 Simon 的博客中搜索相关内容

如果有机会，请优先查阅维基百科。

示例对话：

问题: 法国的首都是什么？
Thought: 我应该去维基百科查一下法国
Action: wikipedia: France
PAUSE

你将会被再次调用，并收到这个：

Observation: France is a country. The capital is Paris.

然后你输出：

Answer: 法国的首都是巴黎
""".strip()


action_re = re.compile('^Action: (\w+): (.*)$')

def query(question, max_turns=5):
    i = 0
    bot = ChatBot(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [action_re.match(a) for a in result.split('\n') if action_re.match(a)]
        if actions:
            # 发现一个需要执行的行动
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception("未知行动: {}: {}".format(action, action_input))
            print(" -- 执行 {} {}".format(action, action_input))
            observation = known_actions[action](action_input)
            print("观察结果:", observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            return


def wikipedia(q):
    return httpx.get("https://en.wikipedia.org/w/api.php", params={
        "action": "query",
        "list": "search",
        "srsearch": q,
        "format": "json"
    }).json()["query"]["search"][0]["snippet"]


def simon_blog_search(q):
    results = httpx.get("https://datasette.simonwillison.net/simonwillisonblog.json", params={
        "sql": """
        select
          blog_entry.title || ': ' || substr(html_strip_tags(blog_entry.body), 0, 1000) as text,
          blog_entry.created
        from
          blog_entry join blog_entry_fts on blog_entry.rowid = blog_entry_fts.rowid
        where
          blog_entry_fts match escape_fts(:q)
        order by
          blog_entry_fts.rank
        limit
          1""".strip(),
        "_shape": "array",
        "q": q,
    }).json()
    return results[0]["text"]

def calculate(what):
    return eval(what)

known_actions = {
    "wikipedia": wikipedia,
    "calculate": calculate,
    "simon_blog_search": simon_blog_search
}


if __name__ == '__main__':
    # 示例问题（来自 Simon Willison 的文章 https://til.simonwillison.net/llms/python-react-pattern）
    examples = [
        ("Wikipedia 查询", "What does England share borders with?"),
        ("博客搜索", "Has Simon been to Madagascar?"),
        ("计算器", "Fifteen * twenty five"),
    ]

    print("=== ReAct Agent 示例 ===\n")
    print("可用示例问题:")
    for i, (label, question) in enumerate(examples):
        print(f"  {i+1}. [{label}] {question}")
    print("  0. 自定义问题")

    try:
        choice = input("\n请选择 (0-3): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n再见!")
        exit()

    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(examples):
            question = examples[idx - 1][1]
        else:
            question = input("请输入你的问题: ")
    elif choice == "":
        question = examples[0][1]
    else:
        question = choice

    print(f"\n> {question}\n")
    query(question)
