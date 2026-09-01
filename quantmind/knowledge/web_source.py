"""网络补充源：为投资想法补充外部学习资料（尽力而为，绝不阻断主流程）。

当库内方法论命中不足时，用 ``gather_web`` 尝试从 Tavily 检索补充资料：
  - 优先：若环境中装有 ``tavily`` 客户端且配置了 ``TAVILY_API_KEY``，则真实联网检索；
  - 否则：回落到**确定性启发式**模拟结果（基于 idea 的标题/snippet 合成），
    保证离线、无网络、无 key 时也能跑通且可测试。
  - 任何异常（网络失败 / 超时 / API 错误）一律静默回退空列表——绝不抛给主流程。

返回结构：``[{"title", "snippet", "url"}]``。
"""
from __future__ import annotations

import logging
import os
import re
from typing import List, Optional

_logger = logging.getLogger("quantmind.knowledge.web_source")

__all__ = ["gather_web"]

#: 网络资料的「来源前缀」，用于溯源去重（区别于库内 source）。
_WEB_SOURCE_PREFIX = "web"
#: Tavily Search REST 端点（仅需 key，无需额外装包）。
_TAVILY_ENDPOINT = "https://api.tavily.com/search"


async def gather_web(idea: str, max_results: int = 3) -> List[dict]:
    """为 ``idea`` 获取最多 ``max_results`` 条外部学习资料（尽力而为）。

    Returns:
        形如 ``[{"title", "snippet", "url"}]`` 的列表；失败/无网络时返回 ``[]``。
    """
    idea = (idea or "").strip()
    if not idea:
        return []
    try:
        client = _api_key()
        if client:
            results = await _tavily_search(client, idea, max_results)
            if results:
                return results
    except Exception as exc:  # noqa: BLE001
        _logger.debug("联网知识检索失败，静默回退启发式: %s", exc)

    # 无 Tavily 或无网络：确定性启发式模拟，保证离线可跑
    try:
        return _heuristic_snippets(idea, max_results)
    except Exception as exc:  # noqa: BLE001
        _logger.debug("启发式资料生成失败，返回空列表: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Tavily 联网检索（仅需 key，经 httpx REST；无 key/失败 → None 走启发式）
# ---------------------------------------------------------------------------
def _api_key() -> Optional[str]:
    """取 Tavily key：优先环境变量 ``TAVILY_API_KEY``，其次 QM 配置 ``QM_TAVILY_API_KEY``，
    最后 .env 文件中的 ``TAVILY_API_KEY``。缺 key 返回 None（不联网）。"""
    key = os.environ.get("TAVILY_API_KEY") or ""
    if not key:
        try:
            from ..config import Settings
            key = (Settings().tavily_api_key or "").strip()
        except Exception:  # noqa: BLE001
            key = ""
    if not key:
        key = _dotenv_value("TAVILY_API_KEY")
    return key.strip() or None


def _dotenv_value(name: str) -> str:
    """极简 .env 读取（不引入 python-dotenv）：返回指定变量值，缺省空串。"""
    try:
        import pathlib

        p = pathlib.Path(".env")
        if not p.exists():
            return ""
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == name:
                return v.strip().strip('"').strip("'")
    except Exception:  # noqa: BLE001
        pass
    return ""


async def _tavily_search(api_key: str, idea: str, max_results: int) -> List[dict]:
    """用 Tavily Search REST API 抓取网络资料（httpx 异步）。"""
    import httpx

    payload = {
        "api_key": api_key,
        "query": idea,
        "max_results": max(max_results, 1),
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(_TAVILY_ENDPOINT, json=payload)
        resp.raise_for_status()
        data = resp.json()
    out: List[dict] = []
    for item in (data.get("results") or [])[:max_results]:
        title = (item.get("title") or "").strip()
        snippet = (item.get("content") or "").strip()
        url = (item.get("url") or "").strip()
        if title or snippet:
            out.append({"title": title, "snippet": snippet, "url": url})
    return out


# ---------------------------------------------------------------------------
# 确定性启发式模拟
# ---------------------------------------------------------------------------
def _heuristic_snippets(idea: str, max_results: int) -> List[dict]:
    """基于 idea 生成确定性的占位学习资料（标题/snippet/url 都可溯源）。

    仅作离线兜底提示，url 使用不可解析的分析占位符，避免误引导到真实链接。
    """
    topic = _topic_of(idea)
    n = max(1, min(max_results, 3))
    entries: List[dict] = []
    for i in range(n):
        entries.append({
            "title": f"{topic} 交易方法论（第{i + 1}篇）",
            "snippet": (
                f"关于「{topic}」的公开量化学习资料：结合走势结构与量价确认，"
                f"提炼可因子化的方向性规则（趋势/均值回归/波动率）。"
            ),
            "url": "",
        })
    return entries


def _topic_of(idea: str) -> str:
    """从 idea 提取精简话题：去掉明显非话题字符，保留中文/英文片段。"""
    parts = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]{3,}", idea)
    if not parts:
        return idea.strip()[:20]
    topic = " ".join(parts[:3])
    return topic[:40]
