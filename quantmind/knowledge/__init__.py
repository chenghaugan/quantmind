"""知识库（knowledge）：量子投研沉淀（因子 / 策略 / 研究日志）。

- :class:`KnowledgeStore`：SQLite 单文件存储 + 轻量关键词检索 + 列表展示。
"""
from __future__ import annotations

from .store import KnowledgeStore

__all__ = [
    "KnowledgeStore",
]
