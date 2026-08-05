"""可搜索的基线因子种子池 + (表达式, IC) 配对持久化（对标 AlphaBench ``factors/`` 与搜索 SFT 数据）。

两个能力：
  1. **种子池**：一组精心挑选、可在面板 DSL（``panel_eval_expression``）上求值的
     基线因子表达式（动量/反转/波动/量价相关/趋势/分布类），可作为 ``cot_search`` /
     ``ea`` / ``tot`` 的 ``seed`` 源，也用于评测/搜索的初始池（对标 Alpha158 的角色）。
  2. **评估配对持久化**：把搜索/批量评估自然产生的「(表达式, 真实 RankIC/IC)」配对
     落库到 SQLite——论文明确指出这是未来 SFT（监督微调）与评测弱标签的宝贵数据源。

用法::

    from quantmind.research import list_seed_pool, FactorPairStore

    # 种子池（每项为 (name, expression)）
    for name, expr in list_seed_pool():
        ...

    # 持久化评估配对
    store = FactorPairStore()
    store.add_pairs([("delta(close,20)", 0.03, 0.02), ...], market="csi300")
    rows = store.load_pairs(market="csi300")
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_logger = logging.getLogger("quantmind.research.seed_pool")

__all__ = [
    "DEFAULT_SEED_POOL",
    "list_seed_pool",
    "FactorPairStore",
]

# 基线因子种子池：(name, expression)。所有表达式都能被面板 DSL 求值。
# 分主题覆盖：趋势/反转/波动/量价/分布/短长差。
DEFAULT_SEED_POOL: List[Tuple[str, str]] = [
    ("momentum_20", "delta(close, 20)"),
    ("momentum_60", "delta(close, 60)"),
    ("momentum_5", "delta(close, 5)"),
    ("mean_reversion_5", "-delta(close, 5)"),
    ("mean_reversion_20", "-delta(close, 20)"),
    ("volatility_10", "std(close, 10)"),
    ("volatility_20", "std(close, 20)"),
    ("volatility_60", "std(close, 60)"),
    ("volume_ratio_5_20", "mean(volume, 5) / mean(volume, 20)"),
    ("volume_ratio_10_30", "mean(volume, 10) / mean(volume, 30)"),
    ("price_volume_corr_10", "corr(close, volume, 10)"),
    ("price_volume_corr_20", "corr(close, volume, 20)"),
    ("close_sma_20", "close - mean(close, 20)"),
    ("zscore_30", "ts_zscore(close, 30)"),
    ("rank_20", "rank(close, 20)"),
    ("range_position", "(close - low) / (high - low)"),
    ("drawdown_norm_25", "(ts_max(close, 25) - close) / mean(volume, 25)"),
    ("log_close_delta", "delta(log(close), 10)"),
    ("signed_volume_1", "sign(delta(close, 1)) * delta(volume, 1)"),
    ("slope_close_10", "slope(close, 10)"),
]


def list_seed_pool() -> List[Tuple[str, str]]:
    """返回按名称排序的种子池（name, expression）。"""
    return list(DEFAULT_SEED_POOL)


class FactorPairStore:
    """(表达式, 真实 IC) 配对持久化，供搜索 / SFT / 评测弱标签使用。

    SQLite 落库，key 为 market|forward_periods|expression；重复写入用最新覆盖。
    线程安全（每操作独立连接）。
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        root = Path(__file__).resolve().parent.parent.parent  # quantmind/
        self.db_path = db_path or str(root / ".factor_cache" / "factor_pairs.sqlite")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._closing() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS factor_pairs (
                    pair_key TEXT PRIMARY KEY,
                    market TEXT NOT NULL,
                    forward_periods INTEGER NOT NULL,
                    expression TEXT NOT NULL,
                    ic REAL,
                    rank_ic REAL,
                    ic_std REAL,
                    ir REAL,
                    n_samples INTEGER,
                    created_at REAL NOT NULL
                )
                """
            )

    def _closing(self):
        """上下文管理器：成功退出时 commit，无论成败都 close（防 Windows 句柄占用/丢提交）。"""
        import contextlib

        @contextlib.contextmanager
        def _cm():
            conn = self._connect()
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

        return _cm()

    def add_pairs(
        self,
        pairs: Sequence[Tuple[object, ...]],
        market: str = "",
        forward_periods: int = 1,
    ) -> int:
        """写入多条评估配对。

        ``pairs`` 每项为 (expression, ic, rank_ic) 或 (expression, ic, rank_ic, ic_std, ir, n_samples)。
        """
        now = time.time()
        rows = []
        for p in pairs:
            expr = p[0]
            ic = _num(p[1]) if len(p) > 1 else None
            rank_ic = _num(p[2]) if len(p) > 2 else None
            ic_std = _num(p[3]) if len(p) > 3 else None
            ir = _num(p[4]) if len(p) > 4 else None
            n_samples = int(p[5]) if len(p) > 5 and p[5] is not None else 0
            key = f"{market}|f{forward_periods}|{expr}"
            rows.append((key, market, forward_periods, expr, ic, rank_ic,
                         ic_std, ir, n_samples, now))
        with self._closing() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO factor_pairs
                (pair_key, market, forward_periods, expression, ic, rank_ic,
                 ic_std, ir, n_samples, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
        return len(rows)

    def load_pairs(self, market: str = "", forward_periods: Optional[int] = None) -> List[dict]:
        """读取评估配对，可按 market / forward_periods 过滤。"""
        sql = "SELECT market, forward_periods, expression, ic, rank_ic, ic_std, ir, n_samples, created_at FROM factor_pairs"
        conds, args = [], []
        if market:
            conds.append("market=?")
            args.append(market)
        if forward_periods is not None:
            conds.append("forward_periods=?")
            args.append(forward_periods)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        with self._closing() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        with self._closing() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM factor_pairs").fetchone()[0])


def _num(x) -> Optional[float]:
    if x is None:
        return None
    try:
        f = float(x)
        return f if f == f else None
    except (TypeError, ValueError):
        return None
