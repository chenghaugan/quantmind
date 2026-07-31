"""AST 安全沙箱：校验 AI 生成的策略/因子代码，禁止危险操作。

禁止项（对应规划「LLM 幻觉防护」）：
  - import 非白名单模块（尤其 os/sys/ subprocess/ socket/ requests/ http 等）
  - 调用 exec / eval / compile / __import__ / open / input
  - 访问 ``__globals__`` / ``__builtins__`` 等危险属性
通过后仍需在隔离命名空间内 exec（本框架回测引擎只加载经校验的类）。
"""
from __future__ import annotations

import ast
import logging
from typing import List, Tuple

_logger = logging.getLogger("quantmind.ai.sandbox")

_ALLOWED_IMPORT_ROOTS = {
    "quantmind",
    "dataclasses",
    "typing",
    "math",
    "datetime",
    "pandas",
    "numpy",
}

_FORBIDDEN_CALLS = {
    "exec", "eval", "compile", "__import__", "open", "input",
    "system", "popen", "subprocess", "os", "socket", "requests",
    "urllib", "http", "threading", "multiprocessing",
}

_FORBIDDEN_ATTRS = {"__globals__", "__builtins__", "__subclasses__", "__bases__"}


class SandboxViolation(Exception):
    """代码违反沙箱规则。"""


def validate_code(source: str) -> Tuple[bool, List[str]]:
    """AST 校验生成代码。返回 (是否通过, 违规说明列表)。"""
    errors: List[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, [f"语法错误: {exc}"]

    for node in ast.walk(tree):
        # import x / import x.y
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in _ALLOWED_IMPORT_ROOTS:
                    errors.append(f"禁止导入模块: {alias.name}")
        # from x import y
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in _ALLOWED_IMPORT_ROOTS:
                errors.append(f"禁止导入模块: {node.module}")
        # 危险调用名
        elif isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in _FORBIDDEN_CALLS:
                errors.append(f"禁止调用: {name}")
        # 危险属性访问
        elif isinstance(node, ast.Attribute):
            if node.attr in _FORBIDDEN_ATTRS:
                errors.append(f"禁止访问属性: {node.attr}")

    return (len(errors) == 0), errors


def compile_strategy(source: str) -> Tuple[bool, str, List[str]]:
    """校验并编译策略源码，返回 (ok, error_msg, errors)。"""
    ok, errors = validate_code(source)
    if not ok:
        return False, "；".join(errors), errors
    try:
        compile(source, "<generated>", "exec")
        return True, "", []
    except SyntaxError as exc:
        return False, str(exc), [f"语法错误: {exc}"]
