"""
企业销售数据智能查询助手（Streamlit 版）

功能：用户输入自然语言问题 -> GLM-4 生成 SQL -> 执行 MySQL 查询
     -> 页面分三块展示：生成的 SQL、查询结果表格、自然语言分析回答。

启动方式：
    streamlit run app.py
"""

import json
import re

import pandas as pd
import pymysql
import streamlit as st
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
    "password": "",
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
        # 建立数据库连接，使用 DictCursor 让查询结果以字典形式返回，便于转成表格
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


@st.cache_resource(show_spinner=False)  # 缓存 Agent，避免每次页面重跑都重新创建
def build_agent(api_key: str) -> AgentExecutor:
    """创建基于 GLM-4 的 SQL Agent。"""
    llm = ChatZhipuAI(
        model="glm-4",  # 也可换成 glm-4-plus / glm-4-flash
        api_key=api_key,
        temperature=0.1,  # 较低温度让 SQL 生成更稳定
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # 注册 execute_sql 工具
    tools = [execute_sql]
    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,                   # 网页端不需要在终端打印调试日志
        return_intermediate_steps=True,  # 保留中间步骤，用于取出 SQL 和查询结果
        max_iterations=5,
        handle_parsing_errors=True,
    )


# ======================== 4. 页面渲染 ========================

st.set_page_config(page_title="企业销售数据智能查询助手", page_icon="📊", layout="wide")
st.title("企业销售数据智能查询助手")
st.caption("输入自然语言问题，助手会自动生成 SQL 并查询 sales_db 数据库。")

# 输入框 + 查询按钮：左侧输入框占 4/5 宽度，右侧按钮占 1/5
col_input, col_button = st.columns([4, 1], vertical_alignment="bottom")
with col_input:
    question = st.text_input(
        "请输入你的问题",
        placeholder="例如：不同省份的销售额是多少？",
    )
with col_button:
    start = st.button("开始查询", type="primary", use_container_width=True)

# 点击按钮时调用 Agent，并把结果存入 session_state（页面刷新后仍能显示）
if start:
    if not question.strip():
        st.warning("请先输入要查询的问题。")
    elif "请在此处填写" in ZHIPU_API_KEY:
        st.error("请先在 app.py 顶部填写 ZHIPU_API_KEY，再开始查询。")
    else:
        with st.spinner("大模型正在生成 SQL 并查询数据库，请稍候..."):
            try:
                result = build_agent(ZHIPU_API_KEY).invoke({"input": question})
                st.session_state.last_result = {"question": question, "result": result}
            except Exception as exc:
                st.session_state.last_result = None
                st.error(f"查询失败：{exc}")

# 展示上一次的查询结果
if st.session_state.get("last_result"):
    data = st.session_state.last_result
    steps = data["result"].get("intermediate_steps", [])

    # 从中间步骤中取出 SQL 和工具返回的原始结果
    generated_sqls = []
    query_observations = []
    for action, observation in steps:
        tool_input = action.tool_input
        sql = tool_input.get("sql_query") if isinstance(tool_input, dict) else str(tool_input)
        generated_sqls.append(sql)
        query_observations.append(observation)

    # ---------- 第一部分：大模型生成的 SQL ----------
    st.subheader("【大模型生成的 SQL 语句】")
    if generated_sqls:
        for sql in generated_sqls:
            st.code(sql, language="sql")
    else:
        st.info("本次回答没有生成 SQL（可能问题与数据查询无关）。")

    # ---------- 第二部分：数据库查询结果 ----------
    st.subheader("【数据库查询结果】")
    if query_observations:
        for observation in query_observations:
            try:
                rows = json.loads(observation)
                if rows:
                    df = pd.DataFrame(rows)  # 查询结果转成表格
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("查询成功，但没有匹配的数据。")
            except (json.JSONDecodeError, TypeError):
                # 不是 JSON，说明 SQL 执行失败，直接把错误信息展示出来
                st.error(observation)
    else:
        st.info("没有调用数据库查询工具。")

    # ---------- 第三部分：大模型的自然语言分析回答 ----------
    st.subheader("【大模型的自然语言分析回答】")
    st.markdown(data["result"].get("output", "（无回答）"))
