"""因子迭代搜索的基础数据结构与离线变异器（对标 AlphaBench ``searcher/``）。

本模块定义搜索运行的数据结构（``SearchStep`` / ``SearchResult``）与一条
**确定性变异器**（``mutate_expressions``）——在离线/无 LLM 时可作为
``search_fn`` 的回落实现，产生可评估的新候选表达式，使 CoT 迭代链路在
无网络时也能端到端跑通与测试。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = [
    "SearchStep",
    "SearchResult",
    "mutate_expressions",
]


@dataclass
class SearchStep:
    """搜索轨迹中的一步（一次候选评估）。"""

    round: int
    expression: str
    rank_ic: float = float("nan")
    ic: float = float("nan")
    improved: bool = False
    is_best: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        r4 = lambda x: round(float(x), 4) if x == x else None  # noqa: E731,T202
        return {
            "round": self.round,
            "expression": self.expression,
            "rank_ic": r4(self.rank_ic),
            "ic": r4(self.ic),
            "improved": self.improved,
            "is_best": self.is_best,
            "note": self.note,
        }


@dataclass
class SearchResult:
    """一次因子迭代搜索的完整结果。"""

    seed: str = ""
    _best_expression: str = ""
    best_rank_ic: float = float("nan")
    best_ic: float = float("nan")
    seed_rank_ic: float = float("nan")
    rounds: int = 0
    improved: bool = False
    history: List[SearchStep] = field(default_factory=list)
    # 防泄漏：搜索期用的指标之外，对最终 best 在独立 val 期的评估
    val_rank_ic: Optional[float] = None
    val_ic: Optional[float] = None

    @property
    def best_expression(self) -> str:
        """best 表达式（默认 seed）。"""
        return self._best_expression or self.seed

    def to_dict(self) -> dict:
        r4 = lambda x: round(float(x), 4) if x == x else None  # noqa: E731,T202
        return {
            "seed": self.seed,
            "best_expression": self.best_expression,
            "best_rank_ic": r4(self.best_rank_ic),
            "best_ic": r4(self.best_ic),
            "seed_rank_ic": r4(self.seed_rank_ic),
            "rounds": self.rounds,
            "improved": self.improved,
            "history": [h.to_dict() for h in self.history],
            "val_rank_ic": r4(self.val_rank_ic) if self.val_rank_ic is not None else None,
            "val_ic": r4(self.val_ic) if self.val_ic is not None else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 确定性变异器（离线 search_fn 回落）
# ─────────────────────────────────────────────────────────────────────────────

# 窗口参数正则：op(..., N)
_WINDOW_RE = re.compile(r"(\b\w+\([^()]*?),?\s*(\d{1,3})\)")
# 变量引用
_VAR_RE = re.compile(r"\$?(\b(?:close|open|high|low|volume|amount)\b)")

_WRAP_OPS = [
    ("ts_zscore", "ts_zscore({expr}, 20)"),
    ("rank", "rank({expr})"),
    ("std", "std({expr}, 10)"),
    ("abs", "abs({expr})"),
    ("sign", "sign({expr})"),
]

_WINDOW_SHIFTS = [-8, -3, 3, 8, 15]
_ALT_OPS = [
    ("mean", "ts_zscore"),
    ("sma", "mean"),
    ("std", "sum"),
    ("log", "power"),
]


def _first_expr_var(expr: str) -> str:
    m = _VAR_RE.search(expr)
    return m.group(1) if m else "close"


def _replace_window(expr: str, delta: int) -> str:
    """把第一个 ……(x, N) … 的窗口 N 平移 delta（找不到则原样返回）。"""
    parts = []

    def _shift(m):
        parts.append(int(m.group(2)) + delta)
        return f"{m.group(1)}, {parts[-1]})"

    out = _WINDOW_RE.sub(_shift, expr, count=1)
    if parts:
        return out
    return expr


def _swap_op(expr: str) -> str:
    """把第一个时序算子的名字替换为替代算子（若识别到）。"""
    for op, repl in _ALT_OPS:
        # 匹配 op(x, N) 且 op 是字首
        pat = re.compile(rf"\b{op}\b")
        if pat.search(expr):
            return pat.sub(repl, expr, count=1)
    return expr


def mutate_expressions(
    expression: str,
    n: int = 6,
    rng: Optional[object] = None,
) -> List[str]:
    """对单个因子表达式做**确定性/半随机**变异，产出 ``n`` 个候选表达式。

    Mutation 操作（按固定顺序，用有限随机偏移避免候选高度重复）：
      1. 平移首个窗口参数（±8/±3/+3/+8/+15）
      2. 替换首个时序算子（mean↔ts_zscore 等）
      3. 包裹算子（ts_zscore / rank / std / abs / sign）
      4. 加减一个基准项（对目标变量做 ts_zscore 或 rank）

    Args:
        expression: 因子表达式（函数式或 Qlib 式）。
        n: 期望候选数量。
        rng: 可选 ``numpy.random.Generator``（用于挑选变异算子）。

    Returns:
        去重后的候选表达式列表（可能少于 ``n``）。
    """
    import numpy as np

    gen = rng or np.random.default_rng(0)
    base = expression
    cands: List[str] = []

    # 1) 窗口偏移
    for d in _WINDOW_SHIFTS:
        cands.append(_replace_window(base, d))
    # 2) 算子替换
    cands.append(_swap_op(base))
    # 3) 包裹
    var = _first_expr_var(base)
    for _, tpl in _WRAP_OPS:
        wrapped = tpl.format(expr=base) if var in base else tpl.format(expr=f"{var}")
        cands.append(wrapped)
    # 4) 加减基准项（略作随机避免全一样）
    for _ in range(3):
        base2 = "rank({v})" if len(cands) % 2 else "ts_zscore({v}, 20)"
        cands.append(f"({base} + {base2.format(v=var)})")

    # 去重、保留原表达式剔除、截断
    seen: List[str] = []
    for c in cands:
        if c and c != base and c not in seen:
            seen.append(c)
    # 若不足 n，用窗口微调补充（-1/-2/+1/+2）
    i = 0
    fill_shift = [-1, -2, 1, 2, -4, 4, -6, 6]
    while len(seen) < n and i < len(fill_shift):
        c = _replace_window(base, fill_shift[i])
        if c and c != base and c not in seen:
            seen.append(c)
        i += 1
    return seen[:n]
