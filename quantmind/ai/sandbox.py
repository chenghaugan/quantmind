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
from typing import List, Optional, Tuple

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
    "getattr", "setattr", "delattr", "vars", "globals", "locals",
    "breakpoint", "exit", "quit", "read_pickle", "load",
}

_FORBIDDEN_ATTRS = {"__globals__", "__builtins__", "__subclasses__", "__bases__"}

# 允许访问的双下划线属性（类定义/继承所需的合法用法）
_ALLOWED_DUNDER_ATTRS = {"__init__", "__name__", "__doc__", "__module__"}

# 裸名黑名单：拦截 `__builtins__[...]`、`__import__` 等不经 Call.func 的逃逸面
_FORBIDDEN_NAMES = {
    "__builtins__", "__import__", "exec", "eval", "compile",
    "open", "input", "getattr", "setattr", "delattr",
    "vars", "globals", "locals", "breakpoint", "exit", "quit",
}

# exec 隔离命名空间中保留的最小内建集（纵深防御）
SAFE_BUILTINS = {
    "len", "range", "min", "max", "abs", "round", "sum", "pow",
    "int", "float", "bool", "str", "list", "dict", "tuple", "set",
    "zip", "enumerate", "sorted", "reversed", "isinstance", "issubclass",
    "hasattr", "map", "filter", "any", "all", "repr", "ValueError", "TypeError",
    "super", "Exception", "RuntimeError", "KeyError", "IndexError", "StopIteration",
    "property", "staticmethod", "classmethod", "divmod", "bytes", "hash",
    "print", "next", "iter", "slice", "callable", "format", "frozenset",
}


def restricted_globals() -> dict:
    """返回供 exec 生成代码使用的受限全局命名空间（最小内建集）。

    包含一个仅放行沙箱白名单根模块的 ``__import__``（如 quantmind/pandas/numpy
    等），其余 import（os/sys/subprocess 等）在执行期被拒绝。
    """
    import builtins

    safe = {name: getattr(builtins, name)
            for name in SAFE_BUILTINS if hasattr(builtins, name)}

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        root = (name or "").split(".")[0]
        if root not in _ALLOWED_IMPORT_ROOTS:
            raise ImportError(f"沙箱禁止导入模块: {name}")
        return builtins.__import__(name, globals, locals, fromlist, level)

    safe["__import__"] = _guarded_import
    # class 语句执行所需（不引入逃逸面）
    safe["__build_class__"] = builtins.__build_class__
    safe["__name__"] = "generated"
    return {"__builtins__": safe}


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
        # 危险调用名（只允许白名单形式的 Name/Attribute 调用）
        elif isinstance(node, ast.Call):
            func = node.func
            if not isinstance(func, (ast.Name, ast.Attribute)):
                errors.append("禁止的调用形式（仅允许直接函数/方法调用）")
                continue
            name = func.id if isinstance(func, ast.Name) else func.attr
            if name in _FORBIDDEN_CALLS:
                errors.append(f"禁止调用: {name}")
        # 危险裸名（拦截 __builtins__ 下标逃逸等）
        elif isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES:
                errors.append(f"禁止引用名称: {node.id}")
        # 危险属性访问：dunder 属性是对象链逃逸的主要通道
        # （().__class__.__base__.__dict__["__loader__"] 等），仅放行合法白名单
        # （如 super().__init__()）；其余双下划线属性一律拒绝
        elif isinstance(node, ast.Attribute):
            if node.attr in _FORBIDDEN_ATTRS or (
                    node.attr.startswith("__") and node.attr.endswith("__")
                    and node.attr not in _ALLOWED_DUNDER_ATTRS):
                errors.append(f"禁止访问属性: {node.attr}")

    return (len(errors) == 0), errors


def compile_strategy(source: str, require_base: Optional[str] = None) -> Tuple[bool, str, List[str]]:
    """校验并编译策略源码，返回 (ok, error_msg, errors)。

    ``require_base`` 非空时额外要求至少一个类的基类名为该值（如 ``CtaTemplate``），
    防止占位/模板代码（如继承 MultiFactorStrategy 的演示类）混入 CTA 策略流程。
    """
    ok, errors = validate_code(source)
    if not ok:
        return False, "；".join(errors), errors
    if require_base:
        has_base = False
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return False, f"语法错误: {exc}", [f"语法错误: {exc}"]
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    name = base.attr if isinstance(base, ast.Attribute) else (
                        base.id if isinstance(base, ast.Name) else "")
                    if name == require_base:
                        has_base = True
                        break
            if has_base:
                break
        if not has_base:
            errors.append(f"未找到继承 {require_base} 的策略类")
    try:
        compile(source, "<generated>", "exec")
        return (len(errors) == 0), "；".join(errors), errors
    except SyntaxError as exc:
        return False, str(exc), [f"语法错误: {exc}"]
