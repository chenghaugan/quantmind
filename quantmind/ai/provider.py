"""可插拔 LLM Provider（对应规划「LLM 可插拔 + Mock 默认」）。

真实接入 DeepSeek/OpenAI/通义/OpenRouter 等 OpenAI 兼容服务时填写 API key 与
Base URL（环境变量 ``QM_LLM_*`` 或「设置」页）。框架默认用 ``MockProvider`` 使
全流程在无 key、无网络时也能跑通演示。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional

import httpx

_logger = logging.getLogger("quantmind.ai.provider")


# --------------------------------------------------------------------------- #
# 错误分类与重试辅助
# --------------------------------------------------------------------------- #
def _extract_error_message(resp: httpx.Response) -> str:
    """从 OpenAI 兼容错误响应体提取可读错误信息（如网关配额提示）。"""
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        return ""
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or "")
        if isinstance(err, str):
            return err
    return ""


def _retry_after_seconds(headers) -> Optional[float]:
    """解析 ``Retry-After`` 头（秒数或 HTTP-date），无法解析返回 None。"""
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        pass
    try:
        dt = parsedate_to_datetime(str(raw))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
    except Exception:  # noqa: BLE001
        return None


def _is_quota_exhausted(message: str) -> bool:
    """判断 429 错误是否为「额度/配额用尽」（用于选择提示语与更长的退避）。"""
    lowered = message.lower()
    return any(k in lowered for k in (
        "quota exceeded", "quota_exceeded", "allocated quota",
        "insufficient quota", "out of quota", "no quota", "额度", "配额",
    ))


def _retry_backoff(attempt: int) -> float:
    """指数退避（秒）：3 / 6 / 12 ...，封顶 30s。"""
    return min(3.0 * (2 ** attempt), 30.0)


class _LLMThrottled(Exception):
    """429 限流/额度不足：可重试（尊重 Retry-After，否则按 base_backoff 指数退避）。

    new-api 等网关的 ``code=throttling`` 通常是短时间窗口限制，重试有较大
    概率恢复；即使是硬额度耗尽，重试也不会造成额外损害。
    """

    def __init__(self, detail: str, retry_after: Optional[float] = None,
                 base_backoff: float = 3.0) -> None:
        super().__init__(detail)
        self.detail = detail
        self.retry_after = retry_after
        self.base_backoff = base_backoff


class LLMProvider(ABC):
    """LLM 提供方抽象。"""

    name: str = "base"

    @abstractmethod
    async def chat(self, system: str, user: str) -> str:
        """返回模型文本回复。"""

    async def chat_messages(self, system: str,
                            messages: List[Dict[str, str]]) -> str:
        """多轮对话：``messages`` = [{"role": "user"|"assistant", "content": ...}]。

        默认实现退化取最后一条 user 消息单轮发送；真 Provider 覆写为完整多轮。
        """
        user = next((m.get("content", "") for m in reversed(messages)
                     if m.get("role") == "user"), "")
        return await self.chat(system, user)

    async def complete(self, prompt: str) -> str:
        return await self.chat("", prompt)


class MockProvider(LLMProvider):
    """离线 Mock：基于关键词启发式，产出与真实 LLM 结构一致的结构化结果。

    让 idea_parser/factor_gen/codegen 在无网络、无 key 时产出**可用**结果。
    """

    name = "mock"

    async def chat(self, system: str, user: str) -> str:
        lowered = user.lower()
        # 研究规格
        if "研究" in user or "research" in lowered or "假设" in user or "idea" in lowered:
            return self._research_json(user)
        # 因子生成
        if "因子" in user or "factor" in lowered:
            return self._factor_json(user)
        # 代码生成
        if "策略" in user or "strategy" in lowered or "代码" in user or "code" in lowered:
            return self._code(user)
        return "（mock）已收到请求。"

    # ---- 启发式构造 ----
    def _research_json(self, idea: str) -> str:
        asset = "期货"
        if any(k in idea for k in ["期权", "option"]):
            asset = "期权"
        elif any(k in idea for k in ["港股", "hk", "腾讯", "美团"]):
            asset = "港股"
        elif any(k in idea for k in ["a股", "股票", "茅台", "股票"]):
            asset = "A股"
        factors = ["momentum_20", "mean_reversion_60", "volatility_20"]
        if "期限" in idea or "term" in idea.lower():
            factors = ["term_structure_20", "open_interest_change_20", "momentum_20"]
        if asset == "期权":
            factors = ["iv_rank", "skew", "momentum_20"]
        out = {
            "asset_class": asset,
            "hypothesis": f"基于「{idea}」，相关因子具有稳定的预测能力（多空分层有效）",
            "suggested_factors": factors,
            "risk_notes": ["前视偏差校验", "样本外验证", "流动性与滑点", "极端行情止损"],
        }
        return json.dumps(out, ensure_ascii=False)

    def _factor_json(self, idea: str) -> str:
        out = {
            "factors": [
                {"name": "momentum_20", "kind": "momentum", "window": 20, "weight": 1.0},
                {"name": "mean_reversion_60", "kind": "mean_reversion", "window": 60, "weight": -0.5},
            ]
        }
        return json.dumps(out, ensure_ascii=False)

    def _code(self, idea: str) -> str:
        return (
            "from quantmind.strategy.multifactor import MultiFactorStrategy\n"
            "from quantmind.research.target import FactorSpec\n\n"
            "class GeneratedStrategy(MultiFactorStrategy):\n"
            "    def __init__(self, context, setting=None):\n"
            "        self.specs = [\n"
            "            FactorSpec(name='momentum_20', kind='momentum', window=20, weight=1.0),\n"
            "            FactorSpec(name='mean_reversion_60', kind='mean_reversion', window=60, weight=-0.5),\n"
            "        ]\n"
            "        self.threshold = 0.3\n"
            "        self.size = 1\n"
            "        self.max_pos = 1.0\n"
            "        super().__init__(context, setting)\n"
        )


class _RealProvider(LLMProvider):
    """真实 Provider：OpenAI 兼容 Chat Completions 协议。

    支持 DeepSeek（https://api.deepseek.com/v1）、OpenAI、通义千问、
    OpenRouter 等所有兼容 ``/v1/chat/completions`` 的服务。纯 ``httpx``
    实现，无额外 SDK 依赖。网络失败时自动回退到 MockProvider。
    """

    name = "openai"

    # 瞬时失败（超时/连接/5xx）自动重试次数：网关排队抖动很常见
    _RETRIES = 1

    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "",
                 temperature: float = 0.7, timeout: float = 60.0) -> None:
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.model = model or "gpt-4o-mini"
        self.temperature = temperature
        self.timeout = timeout
        self._mock_fallback = MockProvider()
        # 最近一次真实调用失败并回退 Mock 的原因；None 表示上次调用真实成功。
        # 策略代码生成等关键路径据此把「静默降级」转为显式失败。
        self.last_fallback_reason: Optional[str] = None

    async def _post_chat_completions(self, headers: Dict[str, str],
                                     payload: Dict[str, Any]) -> str:
        """POST /chat/completions，按错误类别决定是否重试。

        - 429（限流/额度不足）→ 尊重 ``Retry-After``，否则指数退避重试
          （额度用尽退避更长）；重试耗尽后上抛，携带可读提示
        - 5xx / 超时 / 连接错误 → 指数退避重试
        - 其它 4xx（鉴权 / 参数错误）→ 立即上抛
        """
        url = f"{self.base_url}/chat/completions"
        max_attempts = self._RETRIES + 1
        last_exc: Optional[Exception] = None
        for attempt in range(max_attempts):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 429:
                    msg = _extract_error_message(resp)
                    retry_after = _retry_after_seconds(resp.headers)
                    if _is_quota_exhausted(msg):
                        detail = (
                            f"LLM 网关额度用尽/被限流 (429)："
                            f"{msg or 'usage allocated quota exceeded'}。"
                            f"正在自动重试；若持续失败请到网关为模型 {self.model} "
                            f"充值/提额，或在「设置」页切换其它有额度的模型。"
                        )
                        base_backoff = 5.0
                    else:
                        detail = (
                            f"LLM 请求被限流 (429)：{msg or 'rate limit exceeded'}。"
                            f"正在自动重试；请降低请求频率或稍后再试。"
                        )
                        base_backoff = 3.0
                    raise _LLMThrottled(detail, retry_after, base_backoff)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except _LLMThrottled as exc:
                last_exc = exc
                delay = (exc.retry_after
                         if exc.retry_after is not None
                         else min(exc.base_backoff * (2 ** attempt), 60.0))
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code >= 500:
                    delay = (_retry_after_seconds(exc.response.headers)
                             or _retry_backoff(attempt))
                else:
                    raise
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                delay = _retry_backoff(attempt)
            if attempt < max_attempts - 1:
                _logger.warning("LLM 调用失败（第 %d/%d 次），%.1fs 后重试: %s",
                                attempt + 1, max_attempts, delay, last_exc)
                await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    async def chat(self, system: str, user: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        self.last_fallback_reason = None
        try:
            return await self._post_chat_completions(headers, payload)
        except Exception as exc:  # noqa: BLE001
            detail = getattr(exc, "detail", None) or f"{type(exc).__name__}: {exc}"
            self.last_fallback_reason = (
                f"{detail} (model={self.model}, base_url={self.base_url})")
            _logger.warning("真实 LLM 调用失败，回退到 Mock: %s", exc)
            return await self._mock_fallback.chat(system, user)

    async def chat_messages(self, system: str,
                            messages: List[Dict[str, str]]) -> str:
        """多轮对话：完整 messages 数组直发 Chat Completions。"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        msgs = ([{"role": "system", "content": system}] if system else [])
        msgs += [{"role": m.get("role", "user"), "content": m.get("content", "")}
                 for m in messages]
        payload = {"model": self.model, "messages": msgs,
                   "temperature": self.temperature}
        self.last_fallback_reason = None
        try:
            return await self._post_chat_completions(headers, payload)
        except Exception as exc:  # noqa: BLE001
            detail = getattr(exc, "detail", None) or f"{type(exc).__name__}: {exc}"
            self.last_fallback_reason = (
                f"{detail} (model={self.model}, base_url={self.base_url})")
            _logger.warning("真实 LLM 多轮调用失败，回退到 Mock: %s", exc)
            return await self._mock_fallback.chat_messages(system, messages)


def build_provider(name: str = "mock", api_key: str = "", base_url: str = "",
                   model: str = "", temperature: float = 0.7,
                   timeout: float = 120.0, **kwargs) -> LLMProvider:
    """按名称与凭据构造 Provider。

    - ``mock`` 或缺少 api_key 时返回离线 ``MockProvider``。
    - 否则返回走 OpenAI 兼容协议的 ``_RealProvider``（策略代码生成耗时长，
      默认 ``timeout`` 120s；可用 ``QM_LLM_TIMEOUT`` 调整）。
    """
    name = (name or "mock").lower()
    if name == "mock" or not api_key:
        return MockProvider()
    return _RealProvider(api_key, base_url or None, model or "", temperature,
                         timeout=float(timeout))
