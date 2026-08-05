"""面板级因子表达式 DSL（对标 AlphaBench FFO 的 Qlib 面板表达式）。

与 :mod:`quantmind.research.factors.expression`（单标的时序求值）不同，本模块在
**多标的面板**（``Panel``：index=日期，columns=标的）上对一个表达式字符串做
向量化求值，返回与面板对齐的 ``DataFrame``（index=日期，columns=标的）。

支持两种等价语法（LLM 生成更自然的是 Qlib 式，人工书写常用函数式）：

  - Qlib 式：``Mean($close, 5)``、``Rank($close, 20)``、``Corr($close,$volume,10)``
  - 函数式：``mean(close, 5)``、``rank(close, 20)``、``corr(close,volume,10)``

变量（均为面板字段，返回 DataFrame）：``close / open / high / low / volume / amount``。

算子分三类：

  - **时序**（逐标的列上计算，返回同型 DataFrame）：
    ``mean/sma/std/sum/ts_rank/ts_min/ts_max/ts_arg_max/ts_arg_min/ts_product/
    ts_zscore/ts_median/slope/decay_linear/delay/delta/corr/cov``
  - **截面**（每个时间截面跨标的计算）：``rank/cs_rank``（百分位排名）、
    ``cs_zscore``（截面 z-score）
  - **标量**：``sign/abs/log/power/signed_power/scale``

安全性：沿用项目 AST 沙箱理念，通过受限 AST 解释器执行（仅允许白名单函数与
面板变量，禁止任意 Python 调用、属性访问、import），避免 LLM 生成代码带来的风险。
"""
from __future__ import annotations

import ast
import operator
import re
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from .alpha_cs import Panel
from .wq import (
    _rank_cs,
    _delay,
    _delta,
    _corr,
    _cov,
    _ts_min,
    _ts_max,
    _ts_arg_max,
    _ts_rank,
    _signed_power,
    _scale,
    _decay_linear,
    _slope,
    _sma,
    _std,
    _sum,
    _ts_arg_min,
    _ts_product,
    _ts_zscore,
    _ts_median,
)

__all__ = [
    "panel_eval_expression",
    "ExpressionError",
    "list_panel_operators",
]


class ExpressionError(ValueError):
    """面板表达式非法、引用未知变量或未授权算子。"""


# ─────────────────────────────────────────────────────────────────────────────
# AST 二元/一元算子（用于表达式中的算术/比较）
# ─────────────────────────────────────────────────────────────────────────────
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}


def _neg(d: pd.DataFrame) -> pd.DataFrame:
    return -d


def _pos(d: pd.DataFrame) -> pd.DataFrame:
    return d


_UNARY_OPS = {
    ast.USub: _neg,
    ast.UAdd: _pos,
}


# ─────────────────────────────────────────────────────────────────────────────
# 算子实现（返回 DataFrame 或标量）
# ─────────────────────────────────────────────────────────────────────────────
# 时序单序列算子：接受 Series 或 DataFrame，对 DataFrame 逐列向量化
def _ts_mean(x, n: int) -> pd.DataFrame:
    return x.rolling(int(n), min_periods=1).mean()


def _ts_sum(x, n: int) -> pd.DataFrame:
    return x.rolling(int(n), min_periods=1).sum()


def _ts_argmax(x, n: int) -> pd.DataFrame:
    return x.rolling(int(n), min_periods=1).apply(
        lambda a: int(np.argmax(a)), raw=True
    )


def _ts_argmin(x, n: int) -> pd.DataFrame:
    return x.rolling(int(n), min_periods=1).apply(
        lambda a: int(np.argmin(a)), raw=True
    )


def _ts_sumprod(x, n: int) -> pd.DataFrame:
    return x.rolling(int(n), min_periods=1).apply(
        lambda a: float(np.prod(a)), raw=True
    )


def _ts_zs(x, n: int) -> pd.DataFrame:
    m = x.rolling(int(n), min_periods=1).mean()
    sd = x.rolling(int(n), min_periods=1).std().replace(0, np.nan)
    return (x - m) / sd


def _ts_med(x, n: int) -> pd.DataFrame:
    return x.rolling(int(n), min_periods=1).median()


def _cs_zscore(d: pd.DataFrame) -> pd.DataFrame:
    """截面 z-score：每个时间截面（每行）跨标的减均值除标准差。"""
    m = d.mean(axis=1)
    sd = d.std(axis=1).replace(0, np.nan)
    return d.sub(m, axis=0).div(sd, axis=0)


def _pwr(x, a: float) -> pd.DataFrame:
    return x ** float(a)


def _sgn(x) -> pd.DataFrame:
    return np.sign(x)


def _absl(x) -> pd.DataFrame:
    return x.abs()


def _log(x) -> pd.DataFrame:
    return np.log(x.clip(lower=1e-9))


def _sig(x) -> pd.DataFrame:
    return _signed_power(x, 1.0)


def _scal(x, a: float = 1.0, win: int = 250) -> pd.DataFrame:
    return _scale(x, float(a), int(win))


def _corry(a, b, n: int) -> pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        return a.rolling(int(n), min_periods=max(2, int(n) // 2)).corr(b)


def _covy(a, b, n: int) -> pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        return a.rolling(int(n), min_periods=max(2, int(n) // 2)).cov(b)


# 算子注册表：统一名称 -> (实现, 是否单序列/是否截面)
# 时序算子几乎都天然支持 DataFrame（按列），因此直接映射到向量化实现。
_TS_OPS: Dict[str, object] = {
    "mean": _ts_mean,
    "sma": _ts_mean,
    "std": _std,
    "sum": _ts_sum,
    "ts_rank": _ts_rank,
    "ts_min": _ts_min,
    "ts_max": _ts_max,
    "ts_arg_max": _ts_argmax,
    "ts_arg_min": _ts_argmin,
    "ts_product": _ts_sumprod,
    "ts_zscore": _ts_zs,
    "ts_median": _ts_med,
    "delay": _delay,
    "delta": _delta,
    "ts_delta": _delta,
    "slope": _slope,
    "decay_linear": _decay_linear,
    "corr": _corry,
    "cov": _covy,
}

_CS_OPS: Dict[str, object] = {
    "rank": _rank_cs,
    "cs_rank": _rank_cs,
    "cs_zscore": _cs_zscore,
}

_SCALAR_OPS: Dict[str, object] = {
    "sign": _sgn,
    "abs": _absl,
    "log": _log,
    "power": _pwr,
    "signed_power": _sig,
    "scale": _scal,
}

# 参数个数约束（用于快速校验）
_OP_MIN_ARGS: Dict[str, int] = {}
# Qlib 式算子名 -> 规范化名（例如 "Mean" -> "mean"）
_QLIB_ALIAS: Dict[str, str] = {
    "TsMean": "mean",
    "Mean": "mean",
    "Sma": "sma",
    "Std": "std",
    "Sum": "sum",
    "SumProd": "ts_product",
    "TsRank": "ts_rank",
    "Min": "ts_min",
    "Max": "ts_max",
    "TsArgMax": "ts_arg_max",
    "TsArgMin": "ts_arg_min",
    "TsZscore": "ts_zscore",
    "Median": "ts_median",
    "TsMedian": "ts_median",
    "Delay": "delay",
    "Delta": "delta",
    "TsDelta": "delta",
    "Slope": "slope",
    "DecayLinear": "decay_linear",
    "Corr": "corr",
    "Cov": "cov",
    "Rank": "rank",
    "CsRank": "cs_rank",
    "CsZscore": "cs_zscore",
    "Sign": "sign",
    "Abs": "abs",
    "Log": "log",
    "Power": "power",
    "SignedPower": "signed_power",
    "Scale": "scale",
}


def _norm_op(name: str) -> str:
    """把算子名规范化为注册表键：QLib 大驼峰 -> 小写函数名，也接受直接小写。"""
    key = name
    if key in _QLIB_ALIAS:
        key = _QLIB_ALIAS[key]
    lk = key.lower()
    if lk in _TS_OPS or lk in _CS_OPS or lk in _SCALAR_OPS:
        return lk
    # 也允许带 ts_ 前缀的小写（ts_mean == mean）
    return key


# 变量名 -> Panel 字段
_VAR_FIELD: Dict[str, str] = {
    "close": "close",
    "open": "open",
    "high": "high",
    "low": "low",
    "volume": "volume",
    "vol": "volume",
    "amount": "amount",
}


def list_panel_operators() -> List[str]:
    """列出当前面板 DSL 支持的全部算子名（Qlib 式 + 函数式）。"""
    names = (
        list(_TS_OPS.keys())
        + list(_CS_OPS.keys())
        + list(_SCALAR_OPS.keys())
        + list(_QLIB_ALIAS.keys())
    )
    return sorted(set(names))


def _resolve_op(fname: str):
    key = _norm_op(fname)
    if key in _TS_OPS:
        return _TS_OPS[key]
    if key in _CS_OPS:
        return _CS_OPS[key]
    if key in _SCALAR_OPS:
        return _SCALAR_OPS[key]
    raise ExpressionError(f"未授权算子: {fname}")


def _get_panel_field(panel: Panel, field: str) -> pd.DataFrame:
    if field == "amount":
        df = panel.amount
        if df is None or df.empty:
            # 退化为典型价充当成交额近似（与 alpha_cs._vwap 一致）
            return (panel.high + panel.low + panel.close) / 3.0
        return df
    return getattr(panel, field)


def _preprocess_expr(expr: str) -> str:
    """把 Qlib 式表达式翻译为函数式，使 ``ast.parse`` 可解析。

    Qlib 式的 ``$`` 变量前缀与"大驼峰算子名"（``Rank($close,20)`` / ``Mean($close,5)``）
    不是合法 Python，需翻译为函数式（``rank(close,20)`` / ``mean(close,5)``）后再求值。
    函数式表达式（无 ``$``、小写算子）原样保留，故两种语法可混用。
    """
    # 1) 去掉变量前缀 $ （$close -> close, $volume -> volume）
    out = re.sub(r"\$([A-Za-z_]\w*)", r"\1", expr)
    # 2) 把 Qlib 大驼峰算子名替换为函数式名（键均为大驼峰，不与变量名冲突）
    for qname, fname in _QLIB_ALIAS.items():
        out = re.sub(rf"\b{re.escape(qname)}\b", fname, out)
    return out


def panel_eval_expression(expr: str, panel: Panel) -> pd.DataFrame:
    """在面板上安全求值因子表达式，返回与 ``panel`` 对齐的 DataFrame。

    变量：close/open/high/low/volume/amount。算子见 :func:`list_panel_operators`。

    Args:
        expr: 表达式字符串，支持函数式 (``mean(close,5)``) 与 Qlib 式
              (``Mean($close,5)``) 两种语法，可混用。
        panel: 多标的面板（index=日期，columns=标的）。

    Returns:
        与 ``panel`` 同 index/columns 的 ``DataFrame``；结果用 0 填补 NaN。

    Raises:
        ExpressionError: 语法非法、未知变量或未授权算子。
    """
    if panel is None or panel.close is None or panel.close.empty:
        raise ExpressionError("panel 为空")

    env = {name: _get_panel_field(panel, field) for name, field in _VAR_FIELD.items()}
    expr = _preprocess_expr(expr)
    tree = ast.parse(expr, mode="eval")
    result = _eval_node(tree.body, env)

    if isinstance(result, pd.Series):
        result = result.to_frame()  # 单标的场景
    if not isinstance(result, pd.DataFrame):
        result = pd.DataFrame(result, index=panel.dates, columns=panel.symbols)
    # 对齐面板的 index/columns
    result = result.reindex(index=panel.dates, columns=panel.symbols)
    return result.fillna(0.0)


def _eval_node(node: ast.AST, env: Dict[str, object]) -> object:
    """受限 AST 求值：仅允许白名单函数、面板变量、算术/比较与布尔表达式。"""
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, env)
        right = _eval_node(node.right, env)
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ExpressionError(f"不支持的二元运算符: {type(node.op).__name__}")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, env)
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ExpressionError(f"不支持的一元运算符: {type(node.op).__name__}")
        return op(operand)
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, env)
        right = _eval_node(node.right, env)
        op = {
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
        }.get(type(node.ops[0]))
        if op is None:
            raise ExpressionError(f"不支持的比较运算符: {type(node.ops[0]).__name__}")
        return op(left, right)
    if isinstance(node, ast.BoolOp):
        vals = [_eval_node(v, env) for v in node.values]
        if isinstance(node.op, ast.And):
            out = vals[0]
            for v in vals[1:]:
                out = out & v
            return out
        out = vals[0]
        for v in vals[1:]:
            out = out | v
        return out
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("仅允许调用内置算子函数")
        fname = node.func.id
        args = [_eval_node(a, env) for a in node.args]
        return _apply_op(fname, args)
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise ExpressionError(f"未知变量: {node.id}")
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Num):  # 兼容老版本
        return node.n
    raise ExpressionError(f"不支持的语法节点: {type(node).__name__}")


def _apply_op(fname: str, args: List[object]) -> object:
    """调用白名单算子，并做参数个数/类型校验。"""
    func = _resolve_op(fname)
    # 数值常量参数（窗口/权重）直接透传；DataFrame 保留
    try:
        if func in (_corry, _covy):
            if len(args) != 3:
                raise ExpressionError(f"{fname} 需要 3 个参数 (a, b, window)")
            a, b, n = args[0], args[1], args[2]
            return func(a, b, int(n))
        if func is _scal:
            if len(args) < 1 or len(args) > 3:
                raise ExpressionError("scale 需要 1~3 个参数 (x, a=1.0, win=250)")
            return func(*args)
        if func in (_sgn, _absl, _log, _sig):
            if len(args) != 1:
                raise ExpressionError(f"{fname} 需要 1 个参数")
            return func(args[0])
        if func is _pwr:
            if len(args) != 2:
                raise ExpressionError("power 需要 2 个参数 (x, a)")
            return func(args[0], args[1])
        # 其余时序/截面算子：单序列 + window（rank/cs_zscore 无 window）
        # rank 兼容两义：rank(x) 截面；rank(x,n) 时序滚动排名（AlphaBench/QLib 常用）
        if func in (_rank_cs, _cs_zscore):
            if len(args) == 1:
                return func(args[0])
            if len(args) == 2 and func is _rank_cs:
                return _ts_rank(args[0], int(args[1]))
            raise ExpressionError(f"{fname} 需要 1 个参数")
        if len(args) != 2:
            raise ExpressionError(f"{fname} 需要 2 个参数 (x, window)")
        x, n = args
        return func(x, int(n))
    except ExpressionError:
        raise
    except (TypeError, ValueError) as e:  # noqa: BLE001
        raise ExpressionError(f"{fname} 参数错误: {e}") from e
