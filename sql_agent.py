"""
基于 LangChain + 智谱 GLM-4 的 Text-to-SQL 智能业务助手。

整体流程：
    用户自然语言问题 -> GLM-4 结合 System Prompt 中的表结构生成 SQL
    -> 通过 Function Call 调用 execute_sql 工具执行 SQL
    -> 把查询结果交回大模型，由大模型组织成最终回答
"""

import json
import re

import pymysql
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

# ======================== 1. 配置区 ========================

# TODO: 在这里填入你的智谱 AI API Key（https://open.bigmodel.cn/ 控制台获取）
ZHIPU_API_KEY = "请在此处填写你的智谱 API Key"

# 本地 MySQL 数据库连接参数
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "123456",
    "database": "sales_db",
    "charset": "utf8mb4",
    "connect_timeout": 5,
}

# 表结构说明：写进 System Prompt，帮助大模型理解字段含义
SCHEMA_INFO = """
数据库：sales_db
表名：sales（销售明细表）

字段说明：
- month     VARCHAR(10)  月份，格式为 'YYYY-MM'，例如 '2024-01'
- province  VARCHAR(10)  省份，例如 '广东'、'浙江'
- channel   VARCHAR(10)  销售渠道，例如 '线上'、'线下'
- amount    INT          销售额（单位：元）

示例数据：
('2024-01', '广东', '线上', 10000)
('2024-01', '广东', '线下', 5000)
('2024-01', '浙江', '线上', 8000)
"""

# System Prompt：约束大模型的角色、SQL 方言和回答方式
SYSTEM_PROMPT = f"""你是一个经验丰富的数据分析师，负责把用户的自然语言问题转换成 MySQL 查询语句。

你可以使用 execute_sql 工具来执行 SQL 并获得真实的查询结果。

【数据库表结构】
{SCHEMA_INFO}

【工作规则】
1. 只使用上面给出的 sales 表，不要编造不存在的表或字段。
2. 只能生成 SELECT 查询（只读），严禁 INSERT / UPDATE / DELETE / DROP 等写操作。
3. 使用 MySQL 语法；字符串常量使用单引号。
4. 除非用户明确要求查看全部数据，否则查询请加上 LIMIT 100，避免返回过多数据。
5. 必须调用 execute_sql 工具获取真实数据，禁止凭空编造查询结果。
6. 拿到查询结果后，用简洁的中文总结回答用户，必要时说明统计口径（如按省份汇总销售额）。
7. 如果问题涉及的字段无法从表结构中确定，先向用户说明，不要猜测。
"""

# ======================== 2. 定义 Function Call 工具 ========================


@tool
def execute_sql(sql_query: str) -> str:
    """执行一条 MySQL 查询语句并返回结果。

    参数：
        sql_query: 需要执行的 SQL 语句，只允许 SELECT 查询。

    返回：
        查询结果的 JSON 字符串；如果执行失败，返回错误信息。
    """
    # 安全检查：去掉首尾空白后必须以 SELECT 或 WITH 开头，防止误执行写操作
    normalized = sql_query.strip().lower()
    if not re.match(r"^(select|with)\b", normalized):
        return "SQL 执行被拒绝：仅允许执行 SELECT 只读查询。"

    conn = None
    try:
        # 建立数据库连接，使用 DictCursor 让查询结果以字典形式返回，便于序列化
        conn = pymysql.connect(**DB_CONFIG, cursorclass=pymysql.cursors.DictCursor)
        with conn.cursor() as cursor:
            cursor.execute(sql_query)
            rows = cursor.fetchall()

        # 返回 JSON 字符串（Decimal / datetime 等类型统一转成字符串）
        return json.dumps(rows, ensure_ascii=False, default=str)
    except Exception as exc:  # 把数据库错误返回给大模型，方便它修正 SQL
        return f"SQL 执行失败：{exc}"
    finally:
        if conn is not None:
            conn.close()


# ======================== 3. 组装 Agent ========================


def build_agent() -> AgentExecutor:
    """创建基于 GLM-4 的 SQL Agent。"""
    # 初始化智谱 GLM-4 模型；temperature 取较低值，让 SQL 生成更稳定
    llm = ChatZhipuAI(
        model="glm-4",  # 也可换成 glm-4-plus / glm-4-flash
        api_key=ZHIPU_API_KEY,
        temperature=0.1,
    )

    # 提示词模板：system 负责设定规则，human 是用户问题，agent_scratchpad 存放大模型的工具调用过程
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # 把 execute_sql 注册为 Agent 可调用的工具
    tools = [execute_sql]
    agent = create_tool_calling_agent(llm, tools, prompt)

    # AgentExecutor 负责“模型思考 -> 调用工具 -> 把结果交回模型”的循环
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,                # 打印完整的思考与工具调用过程
        return_intermediate_steps=True,  # 返回中间步骤，便于取出生成的 SQL 和查询结果
        max_iterations=5,
        handle_parsing_errors=True,
    )


# ======================== 4. 测试入口 ========================


def main() -> None:
    if "请在此处填写" in ZHIPU_API_KEY:
        print("请先在 sql_agent.py 中填写 ZHIPU_API_KEY，再运行本脚本。")
        return

    agent_executor = build_agent()

    question = "不同省份的销售额是多少？"
    print(f"\n用户问题：{question}\n")

    result = agent_executor.invoke({"input": question})

    # 从中间步骤中提取大模型生成的 SQL 和对应的查询结果
    generated_sqls = []
    query_results = []
    for action, observation in result.get("intermediate_steps", []):
        # action.tool_input 是传给 execute_sql 的参数，取其中的 sql_query
        tool_input = action.tool_input
        sql = tool_input.get("sql_query") if isinstance(tool_input, dict) else str(tool_input)
        generated_sqls.append(sql)
        query_results.append(observation)

    print("=" * 60)
    if generated_sqls:
        print("大模型生成的 SQL：")
        for sql in generated_sqls:
            print(sql)
        print("\nSQL 查询结果：")
        for observation in query_results:
            print(observation)
    else:
        print("本次回答没有调用 execute_sql 工具。")
    print("\n最终回答：")
    print(result["output"])
    print("=" * 60)


if __name__ == "__main__":
    main()
