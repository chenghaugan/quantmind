"""KnowledgeService：知识库（因子 / 策略 / 研究日志 / 方法论）的 API 服务层。

封装 :class:`quantmind.knowledge.KnowledgeStore`，暴露为 FastAPI 可调用的
``ingest`` / ``search`` / ``list`` 方法。检索为轻量关键词（无向量库）。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ...knowledge import KnowledgeStore
from ..schemas import KnowledgeIngestRequest, KnowledgeSearchRequest

_logger = logging.getLogger("quantmind.api")

__all__ = ["KnowledgeService"]


class KnowledgeService:
    """知识库读写服务。"""

    def __init__(self, store: Optional[KnowledgeStore] = None) -> None:
        self.store = store or KnowledgeStore()

    def ingest(self, req: KnowledgeIngestRequest) -> dict:
        """写入一条知识库记录。

        ``req.kind`` ∈ ``factor`` | ``strategy`` | ``research_log`` | ``methodology``，
        ``req.payload`` 为对应字段 dict，返回 ``{"kb_id", "kind", "ok"}``。
        """
        kind = (req.kind or "").strip().lower()
        payload: Dict[str, Any] = req.payload or {}

        if kind == "factor":
            kb_id = self.store.ingest_factor(
                name=payload.get("name", ""),
                expression=payload.get("expression", ""),
                idea=payload.get("idea", ""),
                ic=_num(payload.get("ic")),
                ir=_num(payload.get("ir")),
                status=payload.get("status", "active"),
                symbols=payload.get("symbols"),
                asset_class=payload.get("asset_class", ""),
                market=payload.get("market", ""),
            )
        elif kind == "strategy":
            kb_id = self.store.ingest_strategy(
                code=payload.get("code", ""),
                code_safe=bool(payload.get("code_safe", False)),
                idea=payload.get("idea", ""),
                composite_scheme=payload.get("composite_scheme", ""),
                composite_sharpe=_num(payload.get("composite_sharpe")),
                symbols=payload.get("symbols"),
            )
        elif kind == "research_log":
            kb_id = self.store.ingest_research_log(
                idea=payload.get("idea", ""),
                hypotheses=payload.get("hypotheses"),
                evidence=payload.get("evidence"),
            )
        elif kind == "methodology":
            kb_id = self.store.ingest_methodology(
                title=payload.get("title", ""),
                concept=payload.get("concept", ""),
                summary=payload.get("summary", ""),
                content=payload.get("content", ""),
                source=payload.get("source", ""),
                tags=payload.get("tags"),
            )
        else:
            raise ValueError(
                f"未知知识库类型: {kind!r}"
                "（应为 factor | strategy | research_log | methodology）"
            )

        return {"ok": True, "kb_id": kb_id, "kind": kind}

    def search(self, req: KnowledgeSearchRequest) -> dict:
        """轻量关键词检索，返回 ``{"query", "hits": [...]}``。"""
        hits = self.store.search(query=req.query, top_k=req.top_k, kind=req.kind)
        return {"query": req.query, "hits": hits}

    def list(self, kind: Optional[str] = None, limit: int = 50) -> dict:
        """列表展示（最新在前），返回 ``{"items": [...], "total": n}``。"""
        items: List[dict] = self.store.list_items(kind=kind, limit=limit)
        return {"items": items, "total": len(items)}


def _num(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        f = float(x)
        return f if f == f else None
    except (TypeError, ValueError):
        return None
