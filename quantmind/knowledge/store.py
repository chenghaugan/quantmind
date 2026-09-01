"""KnowledgeStore：量子投研沉淀的轻量知识库（SQLite，单文件 ``quantmind/db/knowledge.db``）。

职责：
  - 沉淀对象：因子（``ingest_factor``）、策略（``ingest_strategy``）、
    研究过程日志（``ingest_research_log``）、交易方法论（``ingest_methodology``）。
  - 轻量关键词检索（``search``）：对冗余 ``text`` 列做查询词分词子串包含计数打分，
    取 top_k，**不引入向量库 / embedding** —— 满足离线、零依赖、可移植。
  - 列表展示（``list_items``）：按 kind 过滤，最新在前。

线程安全：所有写操作包在 ``threading.Lock`` 内；每个操作短连接
（``with sqlite3.connect(db_path) as conn``），防 Windows 句柄占用/丢提交
（风格对齐 ``research.factors.seed_pool.FactorPairStore``）。
"""
from __future__ import annotations

import contextlib
import json
import logging
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schema import ALL_SCHEMAS

_logger = logging.getLogger("quantmind.knowledge")

__all__ = ["KnowledgeStore"]

#: 每张表 → 可供检索的候选字段（value 为字段名到权重，score 加权求和）
_FIELD_MAP: Dict[str, Dict[str, int]] = {
    "factor": {"name": 2, "expression": 2, "idea": 1, "symbols": 1,
               "asset_class": 1, "market": 1, "status": 1},
    "strategy": {"code": 2, "idea": 2, "composite_scheme": 1, "symbols": 1},
    "research_log": {"idea": 3, "hypotheses": 1, "evidence": 1},
    "methodology": {"title": 3, "concept": 2, "summary": 2,
                    "content": 1, "source": 1, "tags": 1},
    "run": {"idea": 3, "algo": 2, "status": 1, "asset_class": 1,
            "market": 1, "composite_scheme": 1},
    "trial": {"expression": 3, "status": 2, "reason": 2, "algo": 1,
              "seed": 1},
    "lifecycle": {"strategy_id": 2, "idea": 2, "state": 1, "source": 1,
                  "status": 1, "reason": 1},
}

#: kind → 表名（供 _load_all / search / list_items 统一映射）
_KIND_TABLE = {
    "factor": "factors",
    "strategy": "strategies",
    "research_log": "research_logs",
    "methodology": "methodology",
    "run": "e2e_runs",
    "trial": "factor_trials",
    "lifecycle": "lifecycle",
}

#: 全部受支持的对象种类
_ALL_KINDS = tuple(_KIND_TABLE.keys())

#: 每张表的主键列名（search/list_items 通用取 id）
_PK_COLUMN: Dict[str, str] = {
    "factor": "kb_id", "strategy": "kb_id", "research_log": "kb_id",
    "methodology": "kb_id", "run": "run_id", "trial": "trial_id",
    "lifecycle": "strategy_id",
}


def _tokenize(text: str) -> List[str]:
    """把查询/文本切成词元：中文保留整串，英文/数字按下划线/空格拆分。"""
    parts = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+", text)
    tokens: List[str] = []
    for p in parts:
        if re.fullmatch(r"[\u4e00-\u9fff]+", p):
            # 中文短语整体作为一个词，再补逐字用于宽松匹配
            tokens.append(p)
            if len(p) > 1:
                tokens.extend(p)
        else:
            tokens.append(p)
    return [t for t in tokens if t and len(t) >= 1]


class KnowledgeStore:
    """知识库存储：因子 / 策略 / 研究日志 / 方法论 的落库 + 关键词检索 + 列表。"""

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path:
            self.db_path = db_path
        else:
            # 默认到仓库根 quantmind/db/knowledge.db（可写）
            root = Path(__file__).resolve().parent.parent.parent  # …/quantmind（仓库根）
            self.db_path = str(root / "db" / "knowledge.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    # -- 连接 / 初始化 -----------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _closing(self):
        """上下文管理器：成功退出 commit，无论成败都 close（防 Windows 句柄占用）。"""

        @contextlib.contextmanager
        def _cm():
            conn = self._connect()
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

        return _cm()

    def _init_schema(self) -> None:
        with self._lock, self._closing() as conn:
            for ddl in ALL_SCHEMAS:
                conn.execute(ddl)
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """轻量列迁移：旧库缺列时补充，不影响已有数据。

        新增列后保持 ``SELECT *`` / ``_row_metadata`` 兼容；缺列行读取为空。
        目前只迁移 methodology.meta（机器可读因子指引）。
        """
        cols = {r[1] for r in conn.execute("PRAGMA table_info(methodology)").fetchall()}
        if "meta" not in cols:
            conn.execute("ALTER TABLE methodology ADD COLUMN meta TEXT")

    # -- 写：因子 ---------------------------------------------------------------
    def ingest_factor(
        self,
        name: str,
        expression: str,
        idea: str = "",
        ic: Optional[float] = None,
        ir: Optional[float] = None,
        status: str = "active",
        symbols: Optional[list] = None,
        asset_class: str = "",
        market: str = "",
    ) -> str:
        """落库一条因子记录，返回 kb_id。"""
        kb_id = self._new_id("f")
        syms = _to_json(symbols)
        text = _join_text({
            "name": name, "expression": expression, "idea": idea,
            "symbols": syms, "asset_class": asset_class,
            "market": market, "status": status,
        })
        row = (kb_id, name, expression, idea, _nan_to_none(ic),
               _nan_to_none(ir), status, syms, asset_class, market,
               time.time(), text)
        with self._lock, self._closing() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO factors
                (kb_id, name, expression, idea, ic, ir, status,
                 symbols, asset_class, market, created_at, text)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                row,
            )
        return kb_id

    # -- 写：策略 ---------------------------------------------------------------
    def ingest_strategy(
        self,
        code: str,
        code_safe: bool,
        idea: str = "",
        composite_scheme: str = "",
        composite_sharpe: Optional[float] = None,
        symbols: Optional[list] = None,
    ) -> str:
        """落库一条策略记录，返回 kb_id。"""
        kb_id = self._new_id("s")
        syms = _to_json(symbols)
        text = _join_text({
            "code": code, "idea": idea, "composite_scheme": composite_scheme,
            "symbols": syms,
        })
        row = (kb_id, code, 1 if code_safe else 0, idea, composite_scheme,
               _nan_to_none(composite_sharpe), syms, time.time(), text)
        with self._lock, self._closing() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO strategies
                (kb_id, code, code_safe, idea, composite_scheme,
                 composite_sharpe, symbols, created_at, text)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                row,
            )
        return kb_id

    def list_mined_strategies(self, limit: int = 200) -> List[dict]:
        """读取研究过程挖掘/沉淀的策略脚本（最新在前），供回测运行池耐重启载入/前端展示。

        返回字段：kb_id / code / idea / composite_sharpe / created_at。
        """
        with self._lock, self._closing() as conn:
            rows = conn.execute(
                "SELECT kb_id, code, idea, composite_sharpe, created_at "
                "FROM strategies ORDER BY created_at DESC LIMIT ?",
                (max(1, limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    # -- 写：研究日志 -----------------------------------------------------------
    def ingest_research_log(
        self,
        idea: str,
        hypotheses: Any = None,
        evidence: Any = None,
    ) -> str:
        """落库一条研究过程日志（假设 + 证据），返回 kb_id。"""
        kb_id = self._new_id("r")
        hyp_json = _to_json(hypotheses) if hypotheses is not None else "[]"
        ev_json = _to_json(evidence) if evidence is not None else "{}"
        text = _join_text({"idea": idea,
                           "hypotheses": _text_of(hypotheses, ["statement", "status"]),
                           "evidence": _text_of(evidence, ["verified_exprs"])})
        row = (kb_id, idea, hyp_json, ev_json, time.time(), text)
        with self._lock, self._closing() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO research_logs
                (kb_id, idea, hypotheses, evidence, created_at, text)
                VALUES (?,?,?,?,?,?)
                """,
                row,
            )
        return kb_id

    # -- 写：交易方法论 -----------------------------------------------------
    def ingest_methodology(
        self,
        title: str,
        concept: str = "",
        summary: str = "",
        content: str = "",
        source: str = "",
        tags: Optional[list] = None,
        meta: Optional[dict] = None,
    ) -> str:
        """落库一条交易方法论记录（理论学习资料），返回 kb_id。

        ``meta`` 为机器可读 JSON 字段（如 ``{"implementable": bool, "kind": str,
        "operator": str, "expr_hint": str, "evidence": str}``），供「方法论知识层」
        判断该方法论能否忠实因子化及如何映射。
        """
        kb_id = self._new_id("m")
        tags_json = _to_json(tags)
        meta_json = _to_json(meta)
        text = _join_text({
            "title": title, "concept": concept, "summary": summary,
            "content": content, "source": source, "tags": tags_json,
            "meta": meta_json,
        })
        row = (kb_id, title, concept, summary, content, source,
               tags_json, meta_json, time.time(), text)
        with self._lock, self._closing() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO methodology
                (kb_id, title, concept, summary, content, source, tags,
                 meta, created_at, text)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                row,
            )
        return kb_id

    def update_methodology_meta(self, title: str, meta: Optional[dict] = None) -> bool:
        """按 ``title`` 回填方法论记录的机器可读 ``meta``（幂等；无匹配行返回 False）。

        用于 ``ensure_seed_data`` 给历史库内种子补齐 ``implementable/kind/operator`` 等字段。
        """
        with self._lock, self._closing() as conn:
            cur = conn.execute(
                "UPDATE methodology SET meta=? WHERE title=?",
                (_to_json(meta) if meta else None, title),
            )
            return cur.rowcount > 0

    # -- 写：端到端挖掘运行 ---------------------------------------------------
    def start_e2e_run(
        self,
        run_id: str,
        idea: str,
        asset_class: str = "",
        market: str = "",
        symbols: Optional[list] = None,
        exchange: str = "",
        interval: str = "",
        algo: str = "",
        rounds: int = 0,
        forward_periods: int = 0,
        status: str = "running",
    ) -> None:
        """开启一次端到端挖掘运行（落库 e2e_runs，composite/brief 暂空）。"""
        syms = _to_json(symbols)
        text = _join_text({
            "run_id": run_id, "idea": idea, "asset_class": asset_class,
            "market": market, "symbols": syms, "exchange": exchange,
            "interval": interval, "algo": algo, "rounds": str(rounds),
            "forward_periods": str(forward_periods), "status": status,
        })
        row = (run_id, idea, asset_class, market, syms, exchange, interval,
               algo, rounds, forward_periods, 0, 0, "", None, None, "",
               status, time.time(), text)
        with self._lock, self._closing() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO e2e_runs
                (run_id, idea, asset_class, market, symbols, exchange, interval,
                 algo, rounds, forward_periods, n_representative,
                 n_verified_hypotheses, composite_scheme, composite_fwd_ic,
                 composite_sharpe, brief, status, created_at, text)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                row,
            )

    def finish_e2e_run(
        self,
        run_id: str,
        n_representative: int = 0,
        n_verified_hypotheses: int = 0,
        composite_scheme: str = "",
        composite_fwd_ic: Optional[float] = None,
        composite_sharpe: Optional[float] = None,
        brief: str = "",
        status: str = "done",
    ) -> None:
        """回填一次挖掘运行的统计与 AI 经验 brief；若 run 不存在则忽略。"""
        with self._lock, self._closing() as conn:
            row = conn.execute(
                "SELECT * FROM e2e_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is None:
                return
            text = _join_text({
                "run_id": run_id, "idea": row["idea"],
                "composite_scheme": composite_scheme,
                "n_representative": str(n_representative),
                "composite_fwd_ic": _str_of(composite_fwd_ic),
                "composite_sharpe": _str_of(composite_sharpe),
                "status": status, "brief": brief,
            })
            conn.execute(
                """
                UPDATE e2e_runs SET
                    n_representative=?,
                    n_verified_hypotheses=?,
                    composite_scheme=?,
                    composite_fwd_ic=?,
                    composite_sharpe=?,
                    brief=?,
                    status=?,
                    text=?
                WHERE run_id=?
                """,
                (n_representative, n_verified_hypotheses, composite_scheme,
                 _nan_to_none(composite_fwd_ic), _nan_to_none(composite_sharpe),
                 brief, status, text, run_id),
            )

    # -- 写：因子试验 ---------------------------------------------------------
    def ingest_factor_trial(
        self,
        run_id: str,
        expression: str,
        algo: str = "",
        seed: str = "",
        train_ic: Optional[float] = None,
        val_ic: Optional[float] = None,
        test_ic: Optional[float] = None,
        test_sharpe: Optional[float] = None,
        test_return: Optional[float] = None,
        test_mdd: Optional[float] = None,
        is_representative: bool = False,
        status: str = "active",
        reason: str = "",
        removed_redundant: Optional[list] = None,
    ) -> str:
        """落库一次因子试验（成功或失败），返回 trial_id。"""
        trial_id = self._new_id("t")
        removed = _to_json(removed_redundant)
        text = _join_text({
            "trial_id": trial_id, "run_id": run_id, "expression": expression,
            "algo": algo, "seed": seed, "status": status, "reason": reason,
        })
        row = (trial_id, run_id, expression, algo, seed,
               _nan_to_none(train_ic), _nan_to_none(val_ic),
               _nan_to_none(test_ic), _nan_to_none(test_sharpe),
               _nan_to_none(test_return), _nan_to_none(test_mdd),
               1 if is_representative else 0, status, reason, removed,
               time.time(), text)
        with self._lock, self._closing() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO factor_trials
                (trial_id, run_id, expression, algo, seed, train_ic, val_ic,
                 test_ic, test_sharpe, test_return, test_mdd,
                 is_representative, status, reason, removed_redundant,
                 created_at, text)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                row,
            )
        return trial_id

    # -- 读：端到端挖掘运行 ---------------------------------------------------
    def list_runs(self, limit: int = 20) -> List[dict]:
        """按创建时间倒序返回挖掘运行摘要（最新在前）。"""
        rows: List[sqlite3.Row] = []
        with self._lock, self._closing() as conn:
            rows = conn.execute(
                "SELECT * FROM e2e_runs ORDER BY created_at DESC LIMIT ?",
                (max(1, limit),),
            ).fetchall()
        return [
            {
                "run_id": r["run_id"], "kind": "run",
                "text": r["text"] or _fallback_text(r),
                "created_at": _iso(r["created_at"]),
                "metadata": _row_metadata(r, "run"),
            }
            for r in rows
        ]

    def get_run(self, run_id: str) -> Optional[dict]:
        """取单条挖掘运行；不存在返回 None。"""
        with self._lock, self._closing() as conn:
            row = conn.execute(
                "SELECT * FROM e2e_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row["run_id"], "kind": "run",
            "text": row["text"] or _fallback_text(row),
            "created_at": _iso(row["created_at"]),
            "metadata": _row_metadata(row, "run"),
        }

    def trials_for_run(self, run_id: str, limit: int = 100) -> List[dict]:
        """返回某次运行下的全部因子试验，created_at 升序。"""
        rows: List[sqlite3.Row] = []
        with self._lock, self._closing() as conn:
            rows = conn.execute(
                """
                SELECT * FROM factor_trials WHERE run_id=?
                ORDER BY created_at ASC LIMIT ?
                """,
                (run_id, max(1, limit)),
            ).fetchall()
        return [
            {
                "trial_id": r["trial_id"], "run_id": r["run_id"],
                "kind": "trial", "text": r["text"] or _fallback_text(r),
                "created_at": _iso(r["created_at"]),
                "metadata": _row_metadata(r, "trial"),
            }
            for r in rows
        ]

    # -- 写/读：策略生命周期 -------------------------------------------------
    def upsert_strategy_lifecycle(
        self,
        strategy_id: str,
        run_id: str = "",
        idea: str = "",
        state: str = "IDEA",
        source: str = "",
        code: str = "",
        code_safe: Optional[bool] = None,
        symbols: Optional[list] = None,
        status: str = "",
        reason: str = "",
        brief: str = "",
    ) -> str:
        """落库/覆盖一条策略生命周期记录（strategy_id 为 PK），返回 strategy_id。

        新增行 history 初始化为 ``[]``；存量行整行替换（含 history/symbols）。
        """
        syms = _to_json(symbols)
        text = _join_text({
            "strategy_id": strategy_id, "idea": idea, "state": state,
            "source": source, "status": status, "reason": reason,
        })
        now = time.time()
        cs = (1 if code_safe else 0) if code_safe is not None else None
        row = (strategy_id, run_id, idea, state, source, code, cs,
               None, None, None, status, reason, brief, "[]", syms,
               now, now, text)
        with self._lock, self._closing() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO lifecycle
                (strategy_id, run_id, idea, state, source, code, code_safe,
                 sharpe, max_drawdown, composite_fwd_ic, status, reason, brief,
                 history, symbols, created_at, updated_at, text)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                row,
            )
        return strategy_id

    def update_strategy_state(
        self,
        strategy_id: str,
        *,
        state: Optional[str] = None,
        sharpe: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        composite_fwd_ic: Optional[float] = None,
        status: Optional[str] = None,
        reason: Optional[str] = None,
        brief: Optional[str] = None,
        code: Optional[str] = None,
        code_safe: Optional[bool] = None,
    ) -> None:
        """部分更新某策略生命周期字段（只改非 None 字段），并刷新 updated_at 与 text。

        行不存在则以缺省值 INSERT 兜底。
        """
        with self._lock, self._closing() as conn:
            row = conn.execute(
                "SELECT * FROM lifecycle WHERE strategy_id=?", (strategy_id,)
            ).fetchone()
        if row is None:
            self.upsert_strategy_lifecycle(
                strategy_id, state=(state or "IDEA"), status=(status or ""),
                reason=(reason or ""), brief=(brief or ""), code=(code or ""),
                code_safe=code_safe)
            # upsert 不落回测指标，单独补齐（保持部分更新语义）
            if (sharpe is not None or max_drawdown is not None
                    or composite_fwd_ic is not None):
                with self._lock, self._closing() as conn:
                    conn.execute(
                        """
                        UPDATE lifecycle SET sharpe=?, max_drawdown=?,
                        composite_fwd_ic=? WHERE strategy_id=?
                        """,
                        (_nan_to_none(sharpe), _nan_to_none(max_drawdown),
                         _nan_to_none(composite_fwd_ic), strategy_id),
                    )
            return

        new_state = state if state is not None else row["state"]
        new_sharpe = sharpe if sharpe is not None else row["sharpe"]
        new_mdd = max_drawdown if max_drawdown is not None else row["max_drawdown"]
        new_fic = composite_fwd_ic if composite_fwd_ic is not None else row["composite_fwd_ic"]
        new_status = status if status is not None else row["status"]
        new_reason = reason if reason is not None else row["reason"]
        new_brief = brief if brief is not None else row["brief"]
        new_code = code if code is not None else row["code"]
        new_cs = (1 if code_safe else 0) if code_safe is not None else row["code_safe"]
        text = _join_text({
            "strategy_id": strategy_id, "idea": row["idea"], "state": new_state,
            "source": row["source"], "status": new_status, "reason": new_reason,
        })
        with self._lock, self._closing() as conn:
            conn.execute(
                """
                UPDATE lifecycle SET
                    state=?, sharpe=?, max_drawdown=?, composite_fwd_ic=?,
                    status=?, reason=?, brief=?, code=?, code_safe=?,
                    updated_at=?, text=?
                WHERE strategy_id=?
                """,
                (new_state, _nan_to_none(new_sharpe), _nan_to_none(new_mdd),
                 _nan_to_none(new_fic), new_status, new_reason, new_brief,
                 new_code, new_cs, time.time(), text, strategy_id),
            )

    def push_strategy_transition(
        self,
        strategy_id: str,
        from_state: str,
        to_state: str,
        note: str = "",
    ) -> None:
        """把一次晋升 {from,to,at,note} 追加到 history 数组末尾并更新 state/updated_at。

        行不存在则先 INSERT 以本次元素为唯一历史；history 解析失败（旧数据）则重建。
        """
        entry = {
            "from": from_state, "to": to_state,
            "at": datetime.now(timezone.utc).isoformat(), "note": note,
        }
        with self._lock, self._closing() as conn:
            row = conn.execute(
                "SELECT * FROM lifecycle WHERE strategy_id=?", (strategy_id,)
            ).fetchone()
            if row is None:
                hist_json = _to_json([entry])
                text = _join_text({"strategy_id": strategy_id})
                now = time.time()
                conn.execute(
                    """
                    INSERT OR REPLACE INTO lifecycle
                    (strategy_id, state, history, created_at, updated_at, text)
                    VALUES (?,?,?,?,?,?)
                    """,
                    (strategy_id, to_state, hist_json, now, now, text),
                )
                return
            hist = _loads(row["history"])
            if not isinstance(hist, list):
                hist = []
            hist.append(entry)
            text = _join_text({
                "strategy_id": strategy_id, "idea": row["idea"], "state": to_state,
                "source": row["source"], "status": row["status"],
                "reason": row["reason"],
            })
            conn.execute(
                """
                UPDATE lifecycle SET history=?, state=?, updated_at=?, text=?
                WHERE strategy_id=?
                """,
                (_to_json(hist), to_state, time.time(), text, strategy_id),
            )

    def get_strategy_lifecycle(self, strategy_id: str) -> Optional[dict]:
        """取单条策略生命周期；不存在返回 None。"""
        with self._lock, self._closing() as conn:
            row = conn.execute(
                "SELECT * FROM lifecycle WHERE strategy_id=?", (strategy_id,)
            ).fetchone()
        if row is None:
            return None
        return self._lifecycle_dict(row)

    def delete_strategy_lifecycle(self, strategy_id: str) -> bool:
        """删除一条策略生命周期记录，返回是否成功删除。"""
        with self._lock, self._closing() as conn:
            cur = conn.execute(
                "DELETE FROM lifecycle WHERE strategy_id = ?", (strategy_id,)
            )
            return cur.rowcount > 0

    def list_strategy_lifecycles(
        self,
        limit: int = 50,
        state: Optional[str] = None,
        source: Optional[str] = None,
    ) -> List[dict]:
        """按最新 updated_at 在前返回策略生命周期列表，可按 state/source 过滤。"""
        clauses: List[str] = []
        params: List[Any] = []
        if state:
            clauses.append("state=?")
            params.append(state)
        if source:
            clauses.append("source=?")
            params.append(source)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._lock, self._closing() as conn:
            rows = conn.execute(
                f"SELECT * FROM lifecycle {where} "
                "ORDER BY updated_at DESC LIMIT ?",  # noqa: S608（表名固定映射）
                params + [max(1, limit)],
            ).fetchall()
        return [self._lifecycle_dict(r) for r in rows]

    def successful_strategies(
        self,
        idea: str = "",
        statuses: tuple = ("verified", "paper", "backtested"),
        top_k: int = 20,
    ) -> List[dict]:
        """检索命中的成功策略，按 sharpe（缺失用 composite_fwd_ic 或 -999）降序。"""
        return self._query_lifecycles(idea, statuses, top_k)

    def failed_strategies(
        self,
        idea: str = "",
        statuses: tuple = ("rejected",),
        top_k: int = 20,
    ) -> List[dict]:
        """检索被拒/失败的策略，供避坑参考。"""
        return self._query_lifecycles(idea, statuses, top_k)

    def _query_lifecycles(
        self, idea: str, statuses: tuple, top_k: int,
    ) -> List[dict]:
        tokens = _tokenize(idea) if idea else []
        status_set = set(statuses or ())
        with self._lock, self._closing() as conn:
            rows = conn.execute("SELECT * FROM lifecycle").fetchall()
        out: List[dict] = []
        for r in rows:
            if status_set and r["status"] not in status_set:
                continue
            # idea 宽松关键词：直接匹配本行 strategy_id / idea / text（lifecycle 自带 idea，
            # 无需 join e2e_runs，避免同类 bug）。
            if tokens and not any(
                t in (r["strategy_id"] or "")
                or t in (r["idea"] or "")
                or (r["text"] or "").find(t) >= 0
                for t in tokens
            ):
                continue
            d = self._lifecycle_dict(r)
            d["kind"] = "lifecycle"
            out.append(d)
        out.sort(
            key=lambda x: (x["sharpe"] if x["sharpe"] is not None
                           else (x["composite_fwd_ic"]
                                 if x["composite_fwd_ic"] is not None else -999)),
            reverse=True,
        )
        return out[: max(1, top_k)]

    def _lifecycle_dict(self, row: sqlite3.Row) -> dict:
        """把 lifecycle 行转成统一读取 dict（get/list/successful/failed 共用）。"""
        return {
            "strategy_id": row["strategy_id"],
            "run_id": row["run_id"],
            "idea": row["idea"],
            "state": row["state"],
            "source": row["source"],
            "status": row["status"],
            "reason": row["reason"],
            "brief": row["brief"],
            "sharpe": _nan_to_none(row["sharpe"]),
            "max_drawdown": _nan_to_none(row["max_drawdown"]),
            "composite_fwd_ic": _nan_to_none(row["composite_fwd_ic"]),
            "history": _loads(row["history"]) or [],
            "symbols": _loads(row["symbols"]) or [],
            "created_at": _iso(row["created_at"]),
            "updated_at": (_iso(row["updated_at"])
                           if row["updated_at"] is not None else None),
            "metadata": _row_metadata(row, "lifecycle"),
        }

    # -- 读：因子成败沉淀 -----------------------------------------------------
    def successful_factors(
        self,
        idea: str = "",
        market: str = "",
        statuses: tuple = ("verified", "passed"),
        top_k: int = 20,
    ) -> List[dict]:
        """跨所有运行检索命中的因子试验，供「复用参考」。

        按 test_ic / train_ic 降序排序；``idea`` 作为宽松关键词过滤
        expression / text。
        """
        return self._query_factors(idea, market, statuses, top_k)

    def failed_factors(
        self,
        idea: str = "",
        market: str = "",
        statuses: tuple = ("rejected", "redundant"),
        top_k: int = 30,
    ) -> List[dict]:
        """跨所有运行检索失败/冗余的因子试验，供「避坑参考」。"""
        return self._query_factors(idea, market, statuses, top_k)

    def _query_factors(
        self,
        idea: str,
        market: str,
        statuses: tuple,
        top_k: int,
    ) -> List[dict]:
        tokens = _tokenize(idea) if idea else []
        market = (market or "").strip()
        rows: List[sqlite3.Row] = []
        with self._lock, self._closing() as conn:
            rows = conn.execute(
                "SELECT * FROM factor_trials"
            ).fetchall()
        status_set = set(statuses or ())
        out: List[dict] = []
        for r in rows:
            if status_set and r["status"] not in status_set:
                continue
            run_meta = self._run_market_offline(r["run_id"])
            run_idea = (run_meta or ("", ""))[0]
            run_market = (run_meta or ("", ""))[1]
            if market and market != run_market:
                continue
            # idea 关键词宽松匹配：命中所属 run 的想法(run_idea)或该 trial 自身
            # 的 expression/text 均可（trial 的 idea 挂在其 run 上，r["text"] 不含 idea）。
            if tokens and not any(
                t in run_idea
                or t in (r["expression"] or "")
                or (r["text"] or "").find(t) >= 0
                for t in tokens
            ):
                continue
            out.append({
                "trial_id": r["trial_id"], "run_id": r["run_id"],
                "kind": "trial", "expression": r["expression"],
                "status": r["status"],
                "test_ic": _nan_to_none(r["test_ic"]),
                "train_ic": _nan_to_none(r["train_ic"]),
                "val_ic": _nan_to_none(r["val_ic"]),
                "test_sharpe": _nan_to_none(r["test_sharpe"]),
                "reason": r["reason"],
                "seed": r["seed"],
                "created_at": _iso(r["created_at"]),
                "metadata": {
                    **{k: v for k, v in _row_metadata(r, "trial").items()
                       if k not in ("run_id",)},
                    "idea": run_idea,
                    "market": run_market,
                    "trial_id": r["trial_id"],
                },
            })
        out.sort(
            key=lambda x: (x["test_ic"] if x["test_ic"] is not None
                           else (x["train_ic"] if x["train_ic"] is not None
                                 else -1.0)),
            reverse=True,
        )
        return out[: max(1, top_k)]

    def _run_market_offline(self, run_id: str) -> Optional[tuple]:
        """按 run_id 取所属运行的 (idea, market)（供成败因子跨运行检索）。

        idea 与 market 必须分开返回：market 过滤只比对 market 字段，
        idea 关键词匹配只对 idea 文本（原先拼在一起会漏检/误检）。
        """
        with self._lock, self._closing() as conn:
            row = conn.execute(
                "SELECT idea, market FROM e2e_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return (row["idea"] or "", row["market"] or "")

    # -- 检索 ---------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 10,
        kind: Optional[str] = None,
    ) -> List[dict]:
        """轻量关键词检索。

        把 ``query`` 分词，对每张表候选行累计「查询词在字段中的包含」得分，
        取 top_k 返回 ``[{"kb_id","kind","text","score","metadata":{...}}]``。
        ``kind`` 可为 ``factor`` / ``strategy`` / ``research_log`` / ``methodology``
        / ``run`` / ``trial`` 精确过滤，不传则跨全部对象一起搜。
        """
        query = (query or "").strip()
        if not query:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []

        kinds = [c for c in _ALL_KINDS
                 if kind is None or c == kind]
        scored: List[dict] = []

        with self._lock, self._closing() as conn:
            for k in kinds:
                rows = self._load_all(conn, k)
                for row in rows:
                    score = _score_row(row, tokens, _FIELD_MAP[k])
                    if score <= 0:
                        continue
                    scored.append({
                        "kb_id": row[_PK_COLUMN[k]],
                        "kind": k,
                        "text": row["text"] or _fallback_text(row),
                        "score": round(score, 3),
                        "metadata": _row_metadata(row, k),
                    })

        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[: max(1, top_k)]

    def bm25_search(
        self,
        query: str,
        top_k: int = 10,
        kind: Optional[str] = None,
    ) -> List[dict]:
        """BM25 语义检索（jieba 分词 + rank_bm25）。

        比关键词子串匹配更鲁棒：支持同义词、部分匹配、词频加权。
        失败时自动降级到 :meth:`search`（关键词子串）。

        依赖：``rank_bm25`` + ``jieba``（可选，缺失则降级）。
        """
        query = (query or "").strip()
        if not query:
            return []

        try:
            from rank_bm25 import BM25Okapi
            import jieba
        except ImportError:
            # 依赖缺失，降级到关键词子串
            return self.search(query, top_k=top_k, kind=kind)

        kinds = [c for c in _ALL_KINDS if kind is None or c == kind]
        candidates: List[tuple] = []  # (row, kind, text)

        with self._lock, self._closing() as conn:
            for k in kinds:
                rows = self._load_all(conn, k)
                for row in rows:
                    text = row["text"] or _fallback_text(row)
                    candidates.append((row, k, text))

        if not candidates:
            return []

        # 构建语料库（jieba 分词）
        corpus = [list(jieba.cut(text)) for _, _, text in candidates]
        bm25 = BM25Okapi(corpus)

        # 查询分词 + 打分
        query_tokens = list(jieba.cut(query))
        scores = bm25.get_scores(query_tokens)

        # 取 top_k
        scored_idx = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:max(1, top_k)]

        results = []
        for idx in scored_idx:
            if scores[idx] <= 0:
                continue
            row, k, text = candidates[idx]
            results.append({
                "kb_id": row[_PK_COLUMN[k]],
                "kind": k,
                "text": text,
                "score": round(float(scores[idx]), 3),
                "metadata": _row_metadata(row, k),
            })

        return results

    def _load_all(self, conn: sqlite3.Connection, kind: str) -> List[sqlite3.Row]:
        table = _KIND_TABLE[kind]
        return conn.execute(
            f"SELECT * FROM {table}"
        ).fetchall()  # noqa: S608（表名来自固定映射，非外部输入）

    # -- 列表 ---------------------------------------------------------------
    def list_items(self, kind: Optional[str] = None, limit: int = 50) -> List[dict]:
        """供应前端列表展示：按创建时间倒序返回对象 dict（含 metadata）。

        缺省不混入 ``run`` / ``trial``（保持原有四类全部列表）；仅当 ``kind``
        显式传 ``run`` / ``trial`` 时返回对应表。
        """
        default_kinds = ("factor", "strategy", "research_log", "methodology")
        kinds = [c for c in default_kinds if kind is None or c == kind]
        if kind and kind not in default_kinds:
            kinds = [kind]
        items: List[dict] = []
        with self._lock, self._closing() as conn:
            for k in kinds:
                rows = self._load_all(conn, k)
                for row in rows:
                    items.append({
                        "kb_id": row[_PK_COLUMN[k]],
                        "kind": k,
                        "text": row["text"] or _fallback_text(row),
                        "created_at": _iso(row["created_at"]),
                        "metadata": _row_metadata(row, k),
                    })
        items.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return items[: max(1, limit)]

    # -- 工具 ---------------------------------------------------------------
    @staticmethod
    def _new_id(prefix: str) -> str:
        return f"{prefix}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"


# =============================================================================
# 内部工具
# =============================================================================
def _to_json(obj: Any) -> str:
    if obj is None:
        return ""
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(obj)


def _nan_to_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        f = float(x)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _join_text(fields: Dict[str, str]) -> str:
    """把候选字段拼接为一段冗余检索文本（空格分隔）。"""
    seen: List[str] = []
    for v in fields.values():
        s = (v or "").strip()
        if s:
            seen.append(s)
    return " ".join(seen)


def _score_row(row: sqlite3.Row, tokens: List[str],
               weight_map: Dict[str, int]) -> int:
    """对单行累计：每个查询词在各加权字段中的命中得分。"""
    score = 0
    for token in tokens:
        for field, weight in weight_map.items():
            val = row[field] if field in row.keys() else ""
            if val and token in str(val):
                score += weight
    return score


def _text_of(obj: Any, keys: List[str]) -> str:
    """从任意结构提取指定字段文本（用于拼接检索冗余文本）。"""
    parts: List[str] = []
    if isinstance(obj, dict):
        for k in keys:
            v = obj.get(k)
            if v is None:
                continue
            if isinstance(v, list):
                parts.extend(str(x) for x in v)
            else:
                parts.append(str(v))
    elif isinstance(obj, list):
        for item in obj:
            parts.append(_text_of(item, keys))
    return " ".join(parts)


def _row_metadata(row: sqlite3.Row, kind: str) -> dict:
    """把某行转成前端可读 metadata dict。"""
    if kind == "factor":
        return {
            "name": row["name"], "expression": row["expression"],
            "idea": row["idea"], "ic": _nan_to_none(row["ic"]),
            "ir": _nan_to_none(row["ir"]), "status": row["status"],
            "symbols": _loads(row["symbols"]), "asset_class": row["asset_class"],
            "market": row["market"],
        }
    if kind == "strategy":
        return {
            "code": row["code"], "code_safe": bool(row["code_safe"]),
            "idea": row["idea"], "composite_scheme": row["composite_scheme"],
            "composite_sharpe": _nan_to_none(row["composite_sharpe"]),
            "symbols": _loads(row["symbols"]),
        }
    if kind == "methodology":
        return {
            "title": row["title"], "concept": row["concept"],
            "summary": row["summary"], "content": row["content"],
            "source": row["source"], "tags": _loads(row["tags"]),
            "meta": _loads(row["meta"]) if "meta" in row.keys() else None,
        }
    if kind == "run":
        return {
            "run_id": row["run_id"], "idea": row["idea"],
            "algo": row["algo"], "status": row["status"],
            "n_representative": row["n_representative"],
            "n_verified_hypotheses": row["n_verified_hypotheses"],
            "composite_scheme": row["composite_scheme"],
            "composite_fwd_ic": _nan_to_none(row["composite_fwd_ic"]),
            "composite_sharpe": _nan_to_none(row["composite_sharpe"]),
            "brief": row["brief"], "created_at": _iso(row["created_at"]),
        }
    if kind == "trial":
        return {
            "run_id": row["run_id"], "expression": row["expression"],
            "status": row["status"], "seed": row["seed"],
            "algo": row["algo"],
            "train_ic": _nan_to_none(row["train_ic"]),
            "val_ic": _nan_to_none(row["val_ic"]),
            "test_ic": _nan_to_none(row["test_ic"]),
            "test_sharpe": _nan_to_none(row["test_sharpe"]),
            "test_return": _nan_to_none(row["test_return"]),
            "test_mdd": _nan_to_none(row["test_mdd"]),
            "is_representative": bool(row["is_representative"]),
            "removed_redundant": _loads(row["removed_redundant"]),
            "reason": row["reason"],
        }
    if kind == "lifecycle":
        return {
            "strategy_id": row["strategy_id"], "run_id": row["run_id"],
            "idea": row["idea"], "state": row["state"], "source": row["source"],
            "status": row["status"], "reason": row["reason"],
            "brief": row["brief"], "sharpe": _nan_to_none(row["sharpe"]),
            "max_drawdown": _nan_to_none(row["max_drawdown"]),
            "composite_fwd_ic": _nan_to_none(row["composite_fwd_ic"]),
            "history": _loads(row["history"]),
            "symbols": _loads(row["symbols"]),
        }
    # research_log
    return {
        "idea": row["idea"],
        "hypotheses": _loads(row["hypotheses"]),
        "evidence": _loads(row["evidence"]),
    }


def _loads(s: Any) -> Any:
    if not s:
        return None
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return s


def _fallback_text(row: sqlite3.Row) -> str:
    meta = _row_metadata(row, _kind_of_table(row))
    return _join_text({k: _str_of(v) for k, v in meta.items()})


def _str_of(v: Any) -> str:
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False, default=str)
    return str(v) if v is not None else ""


def _kind_of_table(row: sqlite3.Row) -> str:
    if "strategy_id" in row.keys() and "state" in row.keys():
        return "lifecycle"
    if "trial_id" in row.keys() and "expression" in row.keys():
        return "trial"
    if "run_id" in row.keys() and "composite_scheme" in row.keys():
        return "run"
    if "name" in row.keys() and "expression" in row.keys():
        return "factor"
    if "code" in row.keys() and "code_safe" in row.keys():
        return "strategy"
    if "title" in row.keys() and "concept" in row.keys():
        return "methodology"
    return "research_log"


def _iso(ts: Any) -> str:
    try:
        import datetime
        return datetime.datetime.fromtimestamp(float(ts)).isoformat()
    except (TypeError, ValueError, OverflowError):
        return str(ts)
