"""一次性回填脚本：把历史端到端跑批结果（老表 factors/strategies）回填成一条
完整的 e2e 闭环记录（e2e_runs / factor_trials），让 Web 端到端流水线页面的
"本次运行 AI 判读 / 历史" 区块立刻有数据可展示。

设计约束：
  - 幂等：若 ``e2e_runs`` 中已存在 run_id == "backfill-历史知识库" 则直接跳过，
    重复运行不会产生重复 run / 重复 trials。
  - 全离线、零依赖（只依赖项目自带 ``KnowledgeStore``）、不发网络、不调 LLM，
    brief 用规则拼接中文。
  - 不改动后端 / API / 既有公开方法逻辑，只新增本脚本。

运行（在仓库根 cwd 下）：
    .venv\\Scripts\\python.exe scripts\\backfill_e2e_history.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

# 把仓库根（父目录 scripts/ 的上一层）加入 sys.path，使
# ``from quantmind.knowledge import KnowledgeStore`` 可导入。
PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from quantmind.knowledge import KnowledgeStore  # noqa: E402

#: 幂等锚点 run_id：历史知识库回填专用。
BACKFILL_RUN_ID = "backfill-历史知识库"

#: 规整为三态：verified → 已验证；其余（active/passed/n/a 等）→ active；不含 rejected。
_VERIFIED_KEYS = {"verified", "passed"}


def _normalize_status(status: str) -> str:
    """把因子 metadata 里的自由状态规整为 verified/active/rejected 三态。"""
    s = (status or "").strip().lower()
    if s in _VERIFIED_KEYS:
        return "verified"
    if s in {"rejected", "redundant"}:
        return "rejected"
    return "active"


def _build_idea_summary(factors: List[dict]) -> str:
    """从因子列表的 idea 字段合并出一个 run 级 idea 概要。

    用首个非空 idea；若存在多个不同 idea 则用"、"合并，最多保留一段不长于
    200 字的描述，避免 run 的 idea 字段过载。
    """
    ideas: List[str] = []
    for f in factors:
        idea = (f.get("metadata") or {}).get("idea") or ""
        idea = idea.strip()
        if idea and idea not in ideas:
            ideas.append(idea)
    if not ideas:
        return "历史知识库沉淀"
    joined = "、".join(ideas)
    return joined[:200]


def _build_composite_scheme(strategies: List[dict]) -> str:
    """从策略 metadata 提取组合方案概要（composite_scheme），取首个非空值。"""
    for s in strategies:
        scheme = (s.get("metadata") or {}).get("composite_scheme") or ""
        if str(scheme).strip():
            return str(scheme).strip()[:500]
    return ""


def _build_composite_sharpe(strategies: List[dict]) -> Optional[float]:
    """从策略 metadata['composite_sharpe'] 取首个非空浮点值；无则 None。"""
    for s in strategies:
        v = (s.get("metadata") or {}).get("composite_sharpe")
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


def _build_brief(n_factors: int, n_strategies: int, n_verified: int) -> str:
    """规则拼接一段中文 AI 经验摘要（离线，无需 LLM）。"""
    return (
        f"本库沉淀了 {n_factors} 个因子、{n_strategies} 个策略，"
        f"其中 {n_verified} 个因子已人工/历史验证为有效。"
        "建议在后续端到端跑批中优先复用这些已验证因子，并对其余未验证因子"
        "补充样本外回测以提升组合稳健性。"
    )


def backfill(store: Optional[KnowledgeStore] = None) -> dict:
    """把历史 knowledge.db 的因子/策略回填成一条 e2e 闭环记录。

    Args:
        store: 可选的 KnowledgeStore；缺省用默认库（仓库根 db/knowledge.db）。
            测试应传入指向临时 DB 的 store，避免污染真实库。

    Returns:
        dict：
            - ``skipped``: 是否因 run 已存在而跳过（幂等）。
            - ``run_id``: 涉及的 run_id。
            - ``n_factors``: 本次处理的因子数（skip 时为已存在 run 的 trials 数）。
            - ``n_verified``: 规整为 verified 的因子数。
            - ``n_strategies``: 读取到的策略数。
            - ``brief``: 写入 run 的 AI 经验摘要。
    """
    ks = store if store is not None else KnowledgeStore()

    # 幂等检查：run 已存在则跳过，不重复回填。
    existing = {r.get("run_id") for r in ks.list_runs(limit=50)}
    if BACKFILL_RUN_ID in existing:
        n_trials = len(ks.trials_for_run(BACKFILL_RUN_ID))
        print(f"[backfill] run '{BACKFILL_RUN_ID}' 已存在，跳过回填 "
              f"（当前 trials={n_trials}）。")
        return {
            "skipped": True,
            "run_id": BACKFILL_RUN_ID,
            "n_factors": n_trials,
            "n_verified": 0,
            "n_strategies": 0,
            "brief": "",
        }

    factors = ks.list_items(kind="factor", limit=500)
    strategies = ks.list_items(kind="strategy", limit=500)

    n_factors = len(factors)
    n_strategies = len(strategies)
    idea = _build_idea_summary(factors)
    composite_scheme = _build_composite_scheme(strategies)
    composite_sharpe = _build_composite_sharpe(strategies)

    # 1) 开启一条 run 概要。
    ks.start_e2e_run(
        run_id=BACKFILL_RUN_ID,
        idea=idea,
        algo="历史知识库回填",
        rounds=n_factors,
    )

    # 2) 逐个因子落 factor_trial。
    n_verified = 0
    for f in factors:
        meta = f.get("metadata") or {}
        expression = (meta.get("expression") or "").strip()
        status = _normalize_status(meta.get("status") or "active")
        if status == "verified":
            n_verified += 1
            is_rep, reason = True, "历史已验证因子"
        else:
            is_rep, reason = False, "历史沉淀因子（未验证）"
        ks.ingest_factor_trial(
            run_id=BACKFILL_RUN_ID,
            expression=expression or meta.get("name") or "",
            algo="历史知识库回填",
            test_ic=meta.get("ic"),
            is_representative=is_rep,
            status=status,
            reason=reason,
        )

    # 3) 回填统计与 AI 经验 brief。
    brief = _build_brief(n_factors, n_strategies, n_verified)
    ks.finish_e2e_run(
        run_id=BACKFILL_RUN_ID,
        n_representative=n_verified,
        n_verified_hypotheses=n_verified,
        composite_scheme=composite_scheme,
        composite_sharpe=composite_sharpe,
        brief=brief,
        status="done",
    )

    print(f"[backfill] 已回填 run '{BACKFILL_RUN_ID}': "
          f"{n_factors} 个因子 / {n_strategies} 个策略 / {n_verified} 个已验证。")
    return {
        "skipped": False,
        "run_id": BACKFILL_RUN_ID,
        "n_factors": n_factors,
        "n_verified": n_verified,
        "n_strategies": n_strategies,
        "brief": brief,
    }


if __name__ == "__main__":
    backfill()
