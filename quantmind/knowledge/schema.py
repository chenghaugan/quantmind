"""SQLite 知识库表结构（内置 sqlite3，零额外依赖）。

三类沉淀对象统一落在 ``quantmind/db/knowledge.db``：
  - ``factors``      单因子（表达式 / IC / IR / 状态 / 归属板块市场）
  - ``strategies``   完整策略（源码 / 安全标志 / 复合alpha Sharpe）
  - ``research_logs`` 研究过程的假设-证据轨迹

每张表都有 ``text`` 冗余列：把可用于关键词检索的字段拼成一段纯文本，
``KnowledgeStore.search`` 对该列做轻量子串分词打分，免去向量库/embedding。
"""
from __future__ import annotations

#: factors：单因子记录（冗余 text 用于检索）
FACTORS_SCHEMA = """
CREATE TABLE IF NOT EXISTS factors (
    kb_id        TEXT PRIMARY KEY,
    name         TEXT,
    expression   TEXT,
    idea         TEXT,
    ic           REAL,
    ir           REAL,
    status       TEXT,
    symbols      TEXT,
    asset_class  TEXT,
    market       TEXT,
    created_at   REAL NOT NULL,
    text         TEXT
)
"""

#: strategies：完整策略记录（冗余 text 用于检索）
STRATEGIES_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    kb_id             TEXT PRIMARY KEY,
    code              TEXT,
    code_safe         INTEGER,
    idea              TEXT,
    composite_scheme  TEXT,
    composite_sharpe  REAL,
    symbols           TEXT,
    created_at        REAL NOT NULL,
    text              TEXT
)
"""

#: research_logs：研究假设-证据轨迹（hypotheses/evidence 存 JSON 文本）
RESEARCH_LOGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_logs (
    kb_id       TEXT PRIMARY KEY,
    idea        TEXT,
    hypotheses  TEXT,
    evidence    TEXT,
    created_at  REAL NOT NULL,
    text        TEXT
)
"""

#: 建表顺序（各表相互独立，可重复执行）
ALL_SCHEMAS = (FACTORS_SCHEMA, STRATEGIES_SCHEMA, RESEARCH_LOGS_SCHEMA)
