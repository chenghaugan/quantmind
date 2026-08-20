"""SQLite 知识库表结构（内置 sqlite3，零额外依赖）。

沉淀对象统一落在 ``quantmind/db/knowledge.db``：
  - ``factors``      单因子（表达式 / IC / IR / 状态 / 归属板块市场）
  - ``strategies``   完整策略（源码 / 安全标志 / 复合alpha Sharpe）
  - ``research_logs`` 研究过程的假设-证据轨迹
  - ``methodology``  交易方法论 / 理论学习资料（缠论 / 威科夫 / 海龟等）

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

#: methodology：交易方法论 / 理论学习资料（title 供幂等去重，text 供关键词检索）
METHODOLOGY_SCHEMA = """
CREATE TABLE IF NOT EXISTS methodology (
    kb_id       TEXT PRIMARY KEY,
    title       TEXT,
    concept     TEXT,
    summary     TEXT,
    content     TEXT,
    source      TEXT,
    tags        TEXT,
    meta        TEXT,
    created_at  REAL NOT NULL,
    text        TEXT
)
"""

#: e2e_runs：一次端到端挖掘运行（idea → 因子挖掘 → 验证 → 复合α）的完整摘要。
#: composite/brief 在 finish_e2e_run 时回填；status 标记 running/done/failed。
E2E_RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS e2e_runs (
    run_id                  TEXT PRIMARY KEY,
    idea                    TEXT,
    asset_class             TEXT,
    market                  TEXT,
    symbols                 TEXT,
    exchange                TEXT,
    interval                TEXT,
    algo                    TEXT,
    rounds                  INTEGER,
    forward_periods         INTEGER,
    n_representative        INTEGER,
    n_verified_hypotheses   INTEGER,
    composite_scheme        TEXT,
    composite_fwd_ic        REAL,
    composite_sharpe        REAL,
    brief                   TEXT,
    status                  TEXT,
    created_at              REAL NOT NULL,
    text                    TEXT
)
"""

#: factor_trials：挖掘中每个因子的一次试验/结果（含成功与失败）。
#: run_id 关联 e2e_runs；removed_redundant 存 JSON 数组（被去重顶掉的表达式）。
FACTOR_TRIALS_SCHEMA = """
CREATE TABLE IF NOT EXISTS factor_trials (
    trial_id            TEXT PRIMARY KEY,
    run_id              TEXT,
    expression          TEXT,
    algo                TEXT,
    seed                TEXT,
    train_ic            REAL,
    val_ic              REAL,
    test_ic             REAL,
    test_sharpe         REAL,
    test_return         REAL,
    test_mdd            REAL,
    is_representative   INTEGER,
    status              TEXT,
    reason              TEXT,
    removed_redundant   TEXT,
    created_at          REAL NOT NULL,
    text                TEXT
)
"""

#: lifecycle：策略生命周期记录（状态机晋升状态 + 策略级经验沉淀）。
#: strategy_id 为 PK（与 LifecycleManager.records 的 key 一致）；run_id 关联 e2e_runs；
#: state 为 LifecycleState.value；history 存晋升历史 JSON 数组；symbols 存 JSON 数组。
#: 冗余 text 由 id/idea/state/source/status/reason 拼成，供关键词检索。
LIFECYCLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS lifecycle (
    strategy_id  TEXT PRIMARY KEY,
    run_id       TEXT,
    idea         TEXT,
    state        TEXT,
    source       TEXT,
    code         TEXT,
    code_safe    INTEGER,
    sharpe       REAL,
    max_drawdown REAL,
    composite_fwd_ic REAL,
    status       TEXT,
    reason       TEXT,
    brief        TEXT,
    history      TEXT,
    symbols      TEXT,
    created_at   REAL NOT NULL,
    updated_at   REAL,
    text         TEXT
)
"""

#: 建表顺序（各表相互独立，可重复执行）
ALL_SCHEMAS = (FACTORS_SCHEMA, STRATEGIES_SCHEMA, RESEARCH_LOGS_SCHEMA,
               METHODOLOGY_SCHEMA, E2E_RUNS_SCHEMA, FACTOR_TRIALS_SCHEMA,
               LIFECYCLE_SCHEMA)
