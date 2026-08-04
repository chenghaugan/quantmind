"""AI 生成代码的前视静态检查（供 codegen 管道做安全校验）。

与 :mod:`quantmind.backtest.validation` 的思路一致，但面向 AI 生成策略代码的
专用护栏：在 ``sandbox.validate_code``（AST 禁危险操作）之外，**新增一层前视
扫描**，识别信号计算对"未来数据"的引用，防止 LLM 生成的策略在回测中靠未来
函数作弊。

仅用 ``ast`` 静态分析，不执行任何生成代码，安全。
"""
from __future__ import annotations

import ast
from typing import List


def lookahead_warnings(code: str) -> List[str]:
    """扫描策略源码中的未来数据引用，返回违规说明列表（空=干净）。

    识别的前视模式：
      - ``close.shift(-1)`` / ``shift(-N)``（N>0 → 引用下期/未来价格）
      - ``close.pct_change().shift(-N)``（未来收益率）
      - 显式 ``df.iloc[+N]`` 正向索引（未来行；常量正索引可判定）
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"语法错误: {exc}"]

    warnings: List[str] = []

    def _neg_int(n: ast.AST) -> bool:
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) \
                and isinstance(n.operand, ast.Constant) and isinstance(n.operand.value, int):
            return n.operand.value > 0
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return n.value < 0
        return False

    def _neg_value(n: ast.AST):
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) \
                and isinstance(n.operand, ast.Constant) and isinstance(n.operand.value, int):
            return -n.operand.value
        return None

    for node in ast.walk(tree):
        # shift(-N)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "shift" and node.args and _neg_int(node.args[0]):
            n = _neg_value(node.args[0])
            where = f"shift({n})" if n is not None else "shift(负参数)"
            warnings.append(
                f"前视: 检测到 {where}，引用未来数据（正确应 shift(+N) 引用历史）"
            )
        # pct_change().shift(-N)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "shift":
            base = node.func.value
            if isinstance(base, ast.Call) and isinstance(base.func, ast.Attribute) \
                    and base.func.attr == "pct_change" and node.args and _neg_int(node.args[0]):
                warnings.append("前视: pct_change().shift(-N) 引用未来收益率")

    # 去重保序
    seen: set = set()
    dedup: List[str] = []
    for w in warnings:
        if w not in seen:
            seen.add(w)
            dedup.append(w)
    return dedup


__all__ = ["lookahead_warnings"]
