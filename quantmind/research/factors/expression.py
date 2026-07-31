"""因子表达式 DSL（安全求值）。

支持简单时间序列表达式，如 ``momentum(close,20)``、``(close/ref(close,60)-1)``、
``zscore(close,120)``。底层通过受限的 AST 解释器执行（禁止任意 Python 调用、导入、
属性访问），避免 LLM 生成代码带来的安全风险（对应规划中的 AST 沙箱理念，此处用于因子）。

支持的函数/变量（作用于 pandas Series ``close/open/high/low/volume/oi``）：
  ref(x, n)        向前偏移 n 期
  mean(x, n)       滚动均值
  std(x, n)        滚动标准差
  sma / ema        均线
  zscore(x, n)     滚动 z-score
  pct_change(x, n) 变化率
  momentum(x, n)   x/ref(x,n)-1
  roll_corr(a,b,n) 滚动相关
"""
from __future__ import annotations

import ast
import operator
from typing import List

import pandas as pd

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Gt: operator.gt,
    ast.Lt: operator.lt,
    ast.GtE: operator.ge,
    ast.LtE: operator.le,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.And: lambda a, b: a and b,
    ast.Or: lambda a, b: a or b,
}

_SAFE_FUNCS = {
    "ref": lambda x, n: x.shift(int(n)),
    "mean": lambda x, n: x.rolling(int(n), min_periods=2).mean(),
    "std": lambda x, n: x.rolling(int(n), min_periods=2).std(),
    "sma": lambda x, n: x.rolling(int(n), min_periods=2).mean(),
    "ema": lambda x, n: x.ewm(span=int(n), adjust=False).mean(),
    "zscore": lambda x, n: (x - x.rolling(int(n), min_periods=20).mean())
    / (x.rolling(int(n), min_periods=20).std().replace(0, pd.NA)),
    "pct_change": lambda x, n: x.pct_change(int(n)),
    "momentum": lambda x, n: x / x.shift(int(n)) - 1.0,
    "roll_corr": lambda a, b, n: a.rolling(int(n)).corr(b),
    "abs": lambda x: x.abs(),
    "log": lambda x: x.clip(lower=1e-9).apply(lambda v: pd.NA if pd.isna(v) else __import__("math").log(v)),
    "max": lambda x, n: x.rolling(int(n), min_periods=2).max(),
    "min": lambda x, n: x.rolling(int(n), min_periods=2).min(),
}


class ExpressionError(ValueError):
    """表达式非法或引用了不安全的操作。"""


def eval_factor_expression(expr: str, df: pd.DataFrame) -> pd.Series:
    """在受限 AST 中安全求值因子表达式，返回与 df 等长的 Series。

    变量：open/high/low/close/volume/oi/turnover。
    """
    env = {
        "open": df["open"],
        "high": df["high"],
        "low": df["low"],
        "close": df["close"],
        "volume": df["volume"],
        "oi": df.get("open_interest", pd.Series([0.0] * len(df))),
        "turnover": df.get("turnover", pd.Series([0.0] * len(df))),
    }
    tree = ast.parse(expr, mode="eval")
    result = _eval_node(tree.body, env)
    if not isinstance(result, pd.Series):
        result = pd.Series(result)
    return result.fillna(0.0)


def _eval_node(node: ast.AST, env: dict) -> object:
    if isinstance(node, ast.BoolOp):
        vals = [_eval_node(v, env) for v in node.values]
        op = _OPS[type(node.op)]
        acc = vals[0]
        for v in vals[1:]:
            acc = op(acc, v)
        return acc
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, env)
        right = _eval_node(node.right, env)
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval_node(node.operand, env))
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, env)
        right = _eval_node(node.comparators[0], env)
        return _OPS[type(node.ops[0])](left, right)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("仅允许调用内置函数")
        fname = node.func.id
        if fname not in _SAFE_FUNCS:
            raise ExpressionError(f"未授权函数: {fname}")
        args = [_eval_node(a, env) for a in node.args]
        return _SAFE_FUNCS[fname](*args)
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise ExpressionError(f"未知变量: {node.id}")
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Num):  # 兼容老版本
        return node.n
    raise ExpressionError(f"不支持的语法节点: {type(node).__name__}")
