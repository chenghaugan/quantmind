"""可插拔 LLM Provider（对应规划「LLM 可插拔 + Mock 默认」）。

真实接入 Anthropic/DeepSeek/OpenAI 时填写 API key（.env 的 QM_LLM_PROVIDER/KEY），
框架默认用 ``MockProvider`` 使全流程在无 key、无网络时也能跑通演示。
"""
from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import List, Optional

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
    """真实 Provider 骨架（Anthropic/DeepSeek/OpenAI 通用）。

    通过 SDK 调用；未安装对应 SDK 时回退 Mock。配置从环境变量读取。
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "") -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    async def chat(self, system: str, user: str) -> str:
        raise NotImplementedError("请安装对应 SDK 并在 ai/provider.py 实现 _RealProvider.chat")


def build_provider(name: str, api_key: str = "", **kwargs) -> LLMProvider:
    """按名称构造 Provider。"""
    name = (name or "mock").lower()
    if name == "mock":
        return MockProvider()
    # 真实 Provider：此处统一回退 Mock（避免强制依赖），并在日志提示
    _logger.warning("未实现真实 Provider %s，回退 Mock", name)
    return MockProvider()
