"""ResearchService: AI 研究（idea -> 因子/策略代码）"""
from typing import Any

from ...ai import ResearchAgent, build_provider
from ..schemas import ResearchRequest, ResearchResult


class ResearchService:
    def __init__(self, provider: Any):
        self.provider = provider

    async def research(self, req: ResearchRequest) -> ResearchResult:
        agent = ResearchAgent(self.provider)
        out = await agent.research(req.idea, req.asset_class or "")
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
        )
