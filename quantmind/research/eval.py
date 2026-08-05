"""因子表达式统一评估入口 + 持久缓存（对标 AlphaBench FFO 的 ``evaluate_factor``）。

把「表达式 → 面板求值 → 截面 IC 评估」封装为单一入口，并对重复评估启用 SQLite
持久缓存（对标 FFO 的 ``factor_cache.sqlite``）。支持单因子评估与批量评估。

典型用法::

    from quantmind.research import (
        panel_eval_expression, evaluate_expression, batch_evaluate_expressions,
        FactorEvalCache,
    )
    from quantmind.research.factors.alpha_cs import Panel

    # 1) 面板求值：表达式 -> DataFrame(date x symbol)
    df = panel_eval_expression("Rank($close, 20)", panel)

    # 2) 统一评估：表达式 -> FactorReport（含 IC / RankIC / ICIR / 衰减…）
    rep = evaluate_expression("Rank($close, 20)", panel, use_cache=True)

    # 3) 批量 + 持久缓存
    reps = batch_evaluate_expressions(
        ["Rank($close,20)", "Mean($volume,5)", "Corr($close,$volume,10)"],
        panel, use_cache=True,
    )
"""
from __future__ import annotations

import os
import json
import sqlite3
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from .evaluator import FactorEvaluator, FactorReport
from .factors.alpha_cs import Panel
from .factors.panel_expr import panel_eval_expression as _panel_eval

__all__ = [
    "evaluate_expression",
    "batch_evaluate_expressions",
    "FactorEvalCache",
]


def _default_cache_path() -> str:
    """默认缓存路径（项目 .factor_cache 目录）。"""
    root = Path(__file__).resolve().parent.parent.parent  # quantmind/
    p = root / ".factor_cache" / "factor_eval.sqlite"
    return str(p)


class FactorEvalCache:
    """因子评估的 SQLite 持久缓存（对标 FFO 的 ``factor_cache.sqlite``）。

    key 由「表达式 + 前向周期」构成；命中时直接反序列化上次的 ``FactorReport``，
    避免对相同表达式重复做昂贵的截面回测。线程安全（每操作独立连接）。
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or _default_cache_path()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS factor_eval (
                    cache_key TEXT PRIMARY KEY,
                    report TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    @staticmethod
    def make_key(expr: str, market: str = "", forward_periods: int = 1) -> str:
        """构造缓存键（表达式 + 市场 + 前向周期）。"""
        return f"{market}|f{forward_periods}|{expr}"

    def get(self, expr: str, market: str = "", forward_periods: int = 1) -> Optional[FactorReport]:
        """读取缓存；未命中返回 None。"""
        key = self.make_key(expr, market, forward_periods)
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT report FROM factor_eval WHERE cache_key=?", (key,)
                ).fetchone()
            if row is None:
                return None
            d = json.loads(row["report"], parse_constant=lambda x: float("nan"))
            return FactorReport(**{k: _from_json_val(v) for k, v in d.items()})
        except Exception:  # noqa: BLE001 缓存损坏不应阻塞评估
            return None

    def set(self, report: FactorReport, expr: str, market: str = "", forward_periods: int = 1) -> None:
        """写入缓存。"""
        key = self.make_key(expr, market, forward_periods)
        payload = json.dumps(report.to_dict(), allow_nan=True, ensure_ascii=False)
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO factor_eval (cache_key, report, created_at) VALUES (?,?,?)",
                    (key, payload, time.time()),
                )
        except Exception:  # noqa: BLE001
            pass

    def clear(self) -> int:
        """清空全部缓存，返回清除条数。"""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM factor_eval")
            return int(cur.rowcount)


def _from_json_val(v):
    """把 JSON 值还原为 FactorReport 字段（nan/None 处理）。"""
    if isinstance(v, float) and (v != v):  # NaN
        return float("nan")
    if isinstance(v, list):
        return [_from_json_val(x) for x in v]
    return v


def _coerce_bars_to_panel(bars_by_symbol) -> Panel:
    """把 {symbol: List[BarData]} 自动构造为 Panel。"""
    if isinstance(bars_by_symbol, Panel):
        return bars_by_symbol
    return Panel.from_bars(bars_by_symbol)


def evaluate_expression(
    expression: str,
    panel,
    forward_periods: int = 1,
    n_groups: int = 5,
    market: str = "",
    use_cache: bool = True,
    cache: Optional[FactorEvalCache] = None,
    factor_name: Optional[str] = None,
) -> FactorReport:
    """对单个表达式做「面板求值 → 截面 IC 评估」，返回 ``FactorReport``。

    Args:
        expression: 因子表达式（函数式或 Qlib 式，见 :mod:`panel_expr`）。
        panel: ``Panel`` 或 ``{symbol: List[BarData]}``（自动构造面板）。
        forward_periods: 前向收益周期数。
        n_groups: 截面分组数。
        market: 市场标识（用于缓存键，如 "csi300"/"rb"）。
        use_cache: 是否启用 SQLite 持久缓存。
        cache: 自定义缓存实例（默认工厂）。
        factor_name: 报告因子名（默认取表达式）。

    Returns:
        截面 IC 评估报告。
    """
    evaluator = FactorEvaluator()
    cache_obj = cache if cache is not None else (FactorEvalCache() if use_cache else None)

    if cache_obj is not None:
        hit = cache_obj.get(expression, market=market, forward_periods=forward_periods)
        if hit is not None:
            return hit

    p = _coerce_bars_to_panel(panel)
    fdf = _panel_eval(expression, p)
    report = evaluator.evaluate_factor_panel(
        fdf, p, forward_periods=forward_periods, n_groups=n_groups,
        factor_name=(factor_name or expression),
    )

    if cache_obj is not None:
        cache_obj.set(report, expression, market=market, forward_periods=forward_periods)
    return report


def batch_evaluate_expressions(
    expressions: List[str],
    panel,
    market: str = "",
    use_cache: bool = True,
    cache: Optional[FactorEvalCache] = None,
    **kwargs,
) -> List[FactorReport]:
    """批量评估多个表达式，返回与输入同序的 ``FactorReport`` 列表。

    依赖已就绪的单因子评估；命中缓存时直接复用，否则求值+评估并回写。
    kwargs（forward_periods/n_groups 等）透传给 :func:`evaluate_expression`。
    """
    return [
        evaluate_expression(expr, panel, market=market, use_cache=use_cache,
                            cache=cache, **kwargs)
        for expr in expressions
    ]
