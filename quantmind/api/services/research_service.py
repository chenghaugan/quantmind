"""ResearchService: AI 研究（idea -> 因子/策略代码）"""
from typing import Any

from ...ai import ResearchAgent, AutoResearchAgent, build_provider
from ..schemas import ResearchRequest, ResearchResult, Provenance


class ResearchService:
    def __init__(self, provider: Any):
        self.provider = provider

    async def research(self, req: ResearchRequest) -> ResearchResult:
        agent = ResearchAgent(self.provider)
        out = await agent.research(req.idea, req.asset_class or "")
        
        # 构建溯源信息（仅包含 ResearchOutput 可用的信息）
        # 如果需要完整的证据链，应使用 AutoResearchAgent
        provenance = Provenance(
            data_sources=["llm_provider"],
            tool_calls=[
                {"step": "parse_idea", "action": "解析投资想法"},
                {"step": "generate_factors", "action": "生成因子"},
                {"step": "generate_strategy_code", "action": "生成策略代码"},
                {"step": "validate_code", "action": "沙箱校验"},
            ],
            evidence_chain=[],
            hypotheses=[],
            research_log=[],
            generated_at=None,
        )
        
        return ResearchResult(
            idea=out.spec.idea,
            asset_class=out.spec.asset_class,
            hypothesis=out.spec.hypothesis,
            suggested_factors=out.spec.suggested_factors,
            risk_notes=out.spec.risk_notes,
            generated_factors=[
                {"name": f.name, "kind": f.kind, "window": f.window, "weight": f.weight}
                for f in out.factors
            ],
            code_safe=out.code_safe,
            code_errors=out.code_errors,
            provenance=provenance,
        )
