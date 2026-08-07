"""可插拔 LLM Provider（对应规划「LLM 可插拔 + Mock 默认」）。

真实接入 DeepSeek/OpenAI/通义/OpenRouter 等 OpenAI 兼容服务时填写 API key 与
Base URL（环境变量 ``QM_LLM_*`` 或「设置」页）。框架默认用 ``MockProvider`` 使
全流程在无 key、无网络时也能跑通演示。
"""
from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import List, Optional

import httpx

_logger = logging.getLogger("quantmind.ai.provider")


class LLMProvider(ABC):
    """LLM 提供方抽象。"""

    name: str = "base"

    @abstractmethod
    async def chat(self, system: str, user: str) -> str:
        """返回模型文本回复。"""

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

    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "",
                 temperature: float = 0.7, timeout: float = 60.0) -> None:
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.model = model or "gpt-4o-mini"
        self.temperature = temperature
        self.timeout = timeout
        self._mock_fallback = MockProvider()

    async def chat(self, system: str, user: str) -> str:
        url = f"{self.base_url}/chat/completions"
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
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            _logger.warning("真实 LLM 调用失败，回退到 Mock: %s", exc)
            return await self._mock_fallback.chat(system, user)


def build_provider(name: str = "mock", api_key: str = "", base_url: str = "",
                   model: str = "", temperature: float = 0.7, **kwargs) -> LLMProvider:
    """按名称与凭据构造 Provider。

    - ``mock`` 或缺少 api_key 时返回离线 ``MockProvider``。
    - 否则返回走 OpenAI 兼容协议的 ``_RealProvider``。
    """
    name = (name or "mock").lower()
    if name == "mock" or not api_key:
        return MockProvider()
    return _RealProvider(api_key, base_url or None, model or "", temperature)
