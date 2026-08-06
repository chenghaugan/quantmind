"""KnowledgeStore：量子投研沉淀的轻量知识库（SQLite，单文件 ``quantmind/db/knowledge.db``）。

职责：
  - 沉淀三类对象：因子（``ingest_factor``）、策略（``ingest_strategy``）、
    研究过程日志（``ingest_research_log``）。
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
    """知识库存储：因子 / 策略 / 研究日志 的落库 + 关键词检索 + 列表。"""

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
        ``kind`` 可为 ``factor`` / ``strategy`` / ``research_log`` 精确过滤，
        不传则跨三种对象一起搜。
        """
        query = (query or "").strip()
        if not query:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []

        kinds = [c for c in ("factor", "strategy", "research_log")
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
                        "kb_id": row["kb_id"],
                        "kind": k,
                    "text": row["text"] or _fallback_text(row),
                    "score": round(score, 3),
                    "metadata": _row_metadata(row, k),
                    })

        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[: max(1, top_k)]

    def _load_all(self, conn: sqlite3.Connection, kind: str) -> List[sqlite3.Row]:
        table = {"factor": "factors", "strategy": "strategies",
                 "research_log": "research_logs"}[kind]
        return conn.execute(
            f"SELECT * FROM {table}"
        ).fetchall()  # noqa: S608（表名来自固定映射，非外部输入）

    # -- 列表 ---------------------------------------------------------------
    def list_items(self, kind: Optional[str] = None, limit: int = 50) -> List[dict]:
        """供应前端列表展示：按创建时间倒序返回对象 dict（含 metadata）。"""
        kinds = [c for c in ("factor", "strategy", "research_log")
                 if kind is None or c == kind]
        items: List[dict] = []
        with self._lock, self._closing() as conn:
            for k in kinds:
                rows = self._load_all(conn, k)
                for row in rows:
                    items.append({
                        "kb_id": row["kb_id"],
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
    if "name" in row.keys() and "expression" in row.keys():
        return "factor"
    if "code" in row.keys() and "code_safe" in row.keys():
        return "strategy"
    return "research_log"


def _iso(ts: Any) -> str:
    try:
        import datetime
        return datetime.datetime.fromtimestamp(float(ts)).isoformat()
    except (TypeError, ValueError, OverflowError):
        return str(ts)
