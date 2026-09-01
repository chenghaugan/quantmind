"""因子挖掘的**端到端流水线**（真实 LLM 驱动，mock 可测）。

把 AlphaBench 建议的多步骤串成一条可复现的流水线，产出「可报告」的结果。
对**真实 LLM provider**（DeepSeek/OpenAI，通过 ``build_provider`` 构造）端到端跑通：

    seed 池
      → 切分 train/val/test（防泄漏）
      → 在 train 期对每个 seed 做迭代搜索（co/ea/tot）→ 收集 best 表达式
      → （可选）judge 用 LLM 对候选池排序/打分
      → 在 train 期对候选池做相关性聚类去冗余 → 每簇保留代表
      → 对每个代表做 IC / val 期指标 / **test 期样本外多空组合回测**
      → 汇总成一份报告（含每因子的 train IC、val 指标、OOS Sharpe/回撤）

设计要点：
  - **防泄漏**：搜索/去重/选代表只用 train 期；val 期仅报告；test 期才是真正的
    样本外回测。与 TA ``PanelSplitter`` / ```cot_search`` 的 val_panel 语义一致。
  - **可插拔 provider**：``provider=None``→Mock（离线可跑）；传真实 provider 即真实 LLM。
  - **纯内存面板操作**：不依赖 DataManager/网络，便于测试与脚本复现；真实数据源
    由调用方先通过 ``DataManager`` 或 ``Panel.from_bars`` 构造好 ``Panel``。

用法::

    from quantmind.research import run_pipeline, PipelineConfig
    from quantmind.ai.provider import build_provider
    from quantmind.research.split import PanelSplitter

    provider = build_provider(name="mock", api_key="")   # 或真实 key
    cfg = PipelineConfig(seeds=["Mean($close,20)", "Rank($close,20)"], algo="co", rounds=4)
    report = run_pipeline(panel, config=cfg, provider=provider)
    print(report["summary"])
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..ai.provider import LLMProvider, MockProvider
from .factors.alpha_cs import Panel
from .split import time_split
from .search import create_algo
from .eval import evaluate_expression
from .dedup import dedup_expressions
from .cross_sectional_backtest import factor_expression_backtest
from .combine import composite_backtest

_logger = logging.getLogger("quantmind.research.pipeline")

__all__ = [
    "PipelineConfig",
    "StepReport",
    "run_pipeline",
]


@dataclass
class PipelineConfig:
    """端到端因子挖掘流水线配置。"""

    seeds: List[str] = field(
        default_factory=lambda: ["Mean($close, 20)", "Rank($close, 20)",
                                 "Corr($close,$volume,10)"])
    algo: str = "co"                       # co | ea | tot
    rounds: int = 4                        # 迭代深度
    forward_periods: int = 1
    train_frac: float = 0.6
    val_frac: float = 0.2
    market: str = ""
    # judge（可选）：True 时对候选池跑 LLM ranking/scoring（需要 provider）
    run_judge: bool = False
    # 去冗余
    dedup_threshold: float = 0.7
    min_abs_ic: float = 0.03              # |rank_ic| 底线（低于 0.03 视为噪声丢弃）
    # 回测
    n_groups: int = 5
    long_short: bool = True
    cost_rate: float = 0.0
    # 净成本/换手宽松闸门（因子侧）：在研究展示 gross 之外，
    # 用换手感知 net 指标过滤高换手/成本吃光的因子，避免其进入复合组合。
    net_gate: bool = False                    # True 才启用该闸门
    max_turnover: float = 0.0                 # 单期平均换手上限（0=关闭）
    min_net_sharpe: float = 0.0               # 净 Sharpe 下限（0=关闭）
    # 复合组合（可选）：把代表因子合成一个可交易 alpha 组合
    run_composite: bool = False          # True 时在 test 期对代表因子做复合回测
    composite_scheme: str = "icir"       # equal | icir | inv_var | min_var
    composite_standardize: str = "zscore"
    # 控制
    max_candidates: int = 12               # 去重后最多回测的代表数
    persist_pairs: bool = True             # 把 (expr, IC) 落库（FactorPairStore）
    # 持续学习闭环：历史知识库上下文（success/fail/briefs）注入搜索 LLM prompt
    knowledge_context: Optional[dict] = None


@dataclass
class StepReport:
    """流水线中单个因子的各期指标。"""

    expression: str = ""
    algo: str = ""
    seed: str = ""
    train_ic: float = float("nan")         # train 期 IC（搜索/选择依据）
    train_rank_ic: float = float("nan")
    val_ic: Optional[float] = None         # val 期 IC（仅报告）
    test_ic: Optional[float] = None        # test 期 IC
    test_sharpe: Optional[float] = None    # test 期 OOS 多空组合 Sharpe（gross，研究展示）
    test_return: Optional[float] = None
    test_mdd: Optional[float] = None
    # 换手感知净成本指标（门槛用）
    test_net_sharpe: Optional[float] = None
    turnover_mean: Optional[float] = None
    cost_ratio: Optional[float] = None
    high_turnover: bool = False            # 换手超上限被闸门标记
    removed_redundant: List[str] = field(default_factory=list)
    daily_returns: List[Optional[float]] = field(default_factory=list)   # test 期多空日收益
    ic_series: List[Optional[float]] = field(default_factory=list)       # test 期逐日截面 IC

    def to_dict(self) -> dict:
        r4 = lambda x: round(float(x), 4) if x == x else None  # noqa: E731,T202
        return {
            "expression": self.expression,
            "algo": self.algo,
            "seed": self.seed,
            "train_ic": r4(self.train_ic),
            "train_rank_ic": r4(self.train_rank_ic),
            "val_ic": r4(self.val_ic) if self.val_ic is not None else None,
            "test_ic": r4(self.test_ic) if self.test_ic is not None else None,
            "test_sharpe": r4(self.test_sharpe) if self.test_sharpe is not None else None,
            "test_return": r4(self.test_return) if self.test_return is not None else None,
            "test_mdd": r4(self.test_mdd) if self.test_mdd is not None else None,
            "test_net_sharpe": r4(self.test_net_sharpe) if self.test_net_sharpe is not None else None,
            "turnover_mean": r4(self.turnover_mean) if self.turnover_mean is not None else None,
            "cost_ratio": r4(self.cost_ratio) if self.cost_ratio is not None else None,
            "high_turnover": self.high_turnover,
            "removed_redundant": self.removed_redundant,
            "daily_returns": self.daily_returns,
            "ic_series": self.ic_series,
        }


def _eval_ic(expr: str, panel: Panel, forward_periods: int, market: str, val_panel=None):
    """在给定面板上评估 IC；可选在 val_panel 上再评估。返回 (train_dict, val_dict)。"""
    train: Dict[str, float] = {}
    val: Dict[str, float] = {}
    try:
        rep = evaluate_expression(expr, panel, forward_periods=forward_periods,
                                  market=market, use_cache=False)
        # ic=pearson（与 search 模块口径一致），rank_ic=rep.ic_mean（spearman）
        _pic = getattr(rep, "ic_pearson", None)
        train["ic"] = _pic if _pic == _pic else float("nan")
        train["rank_ic"] = rep.ic_mean if rep.ic_mean == rep.ic_mean else float("nan")
    except Exception as exc:  # noqa: BLE001
        _logger.debug("train IC 评估失败 %s: %s", expr, exc)
        train = {"ic": float("nan"), "rank_ic": float("nan")}
    if val_panel is not None and len(val_panel.dates):
        try:
            r2 = evaluate_expression(expr, val_panel, forward_periods=forward_periods,
                                     market=market, use_cache=False)
            val["ic"] = r2.ic_mean if r2.ic_mean == r2.ic_mean else None
        except Exception:  # noqa: BLE001
            val = {"ic": None}
    return train, val


def _daily_ic_series(expr: str, panel: Panel, forward_periods: int) -> List[Optional[float]]:
    """计算因子表达式在面板上**逐日截面 rank IC**（供前端画 IC 时序）。

    对每个日期截面：因子值 vs 未来收益做 Spearman（用 pandas，不引入 scipy）；
    标的不足 2 个的截面记为 None。返回与 ``panel.dates`` 等长的列表（缺失为 None）。
    """
    from .factors.panel_expr import panel_eval_expression

    try:
        factor_df = panel_eval_expression(expr, panel)
    except Exception:  # noqa: BLE001
        return []
    if factor_df is None or factor_df.empty:
        return []

    close = panel.close
    fwd = close.pct_change(forward_periods).shift(-forward_periods)
    series: List[Optional[float]] = []
    for d in panel.dates:
        if d not in factor_df.index or d not in fwd.index:
            series.append(None)
            continue
        f = factor_df.loc[d].dropna()
        r = fwd.loc[d].dropna()
        common = f.index.intersection(r.index)
        if len(common) < 2:
            series.append(None)
            continue
        try:
            corr = f[common].astype(float).corr(r[common].astype(float), method="spearman")
            series.append(float(corr))
        except Exception:  # noqa: BLE001
            series.append(None)
    return series


def _run_judge(pool: List[str], provider: LLMProvider) -> Dict[str, float]:
    """（可选）用 LLM 对候选池打分，返回 expr → 分数（供排序；失败则全 0）。"""
    import asyncio
    from .judge import judge_scoring

    async def _score_all() -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for e in pool:
            try:
                r = await judge_scoring(e, provider)
                d = r.prediction if isinstance(r.prediction, dict) else {}
                scores[e] = float(sum(int(v) for v in d.values())) if d else 0.0
            except Exception:  # noqa: BLE001
                scores[e] = 0.0
        return scores

    try:
        return asyncio.run(_score_all())
    except Exception:  # noqa: BLE001
        return {e: 0.0 for e in pool}


def _attach_composite_contribution(composite: Dict[str, object], steps: List[Dict[str, object]]) -> None:
    """为复合组合结果附加「成分风险/收益归因」（proxy）。

    精确的横截面组合无法做严格可加分解，这里给出**近似贡献**：
    ``contribution = weight × 该因子的样本外总收益(test_return)``，
    ``abs_pct = |weight| × |test_return| 在所有成分中的占比``（衡量相对影响）。
    同时带上每个成分的权重 / OOS Sharpe / IC / 收益，供前端「风险归因」表展示。
    """
    weights = composite.get("weights") or {}
    step_map = {s.get("expression"): s for s in steps if s.get("expression")}
    rows: List[Dict[str, object]] = []
    for expr, w in weights.items():
        s = step_map.get(expr) or {}
        _tr = s.get("test_return")
        _w = float(w)
        contrib = (_w * _tr) if (isinstance(_tr, (int, float)) and _tr == _tr) else None
        rows.append({
            "expression": expr,
            "weight": round(_w, 4),
            "test_ic": s.get("test_ic"),
            "test_sharpe": s.get("test_sharpe"),
            "test_return": _tr,
            "contribution": round(contrib, 4) if contrib is not None else None,
        })
    total_abs = sum(abs(r["contribution"]) for r in rows
                    if r.get("contribution") is not None) or 0.0
    if total_abs > 0:
        for r in rows:
            if r.get("contribution") is not None:
                r["abs_pct"] = round(abs(r["contribution"]) / total_abs, 4)
            else:
                r["abs_pct"] = None
    else:
        for r in rows:
            r["abs_pct"] = None
    composite["contribution"] = rows


def run_pipeline(
    panel: Panel,
    config: Optional[PipelineConfig] = None,
    provider: Optional[LLMProvider] = None,
    val_panel: Optional[Panel] = None,
    test_panel: Optional[Panel] = None,
) -> Dict[str, object]:
    """运行端到端因子挖掘流水线。

    Args:
        panel: 训练期面板（搜索/选择依据）。若同时传 train 用,可先自行切分。
        config: 流水线配置。
        provider: LLM provider；None → Mock（离线可跑/可测）。
        val_panel: 可选验证期面板（仅报告指标）。
        test_panel: 可选测试期面板（OOS 多空组合回测）。

    Returns:
        ``{"config": {...}, "steps": [StepReport.to_dict()...], "summary": {...}}``。
    """
    config = config or PipelineConfig()
    provider = provider or MockProvider()
    if panel is None or panel.close.empty:
        raise ValueError("训练期面板为空")

    # 1) 若未显式提供 val/test，按比例自动切分（防泄漏）。
    # 只传 val 或只传 test 时也必须对剩余段做自动切分，否则 train_p=全量 panel，
    # 搜索/去重/打分会看到 "OOS" 期数据，指标实为样本内。
    if val_panel is None or test_panel is None:
        auto_train, auto_val, auto_test = time_split(panel, config.train_frac, config.val_frac)
        train_p = auto_train
        val_p = val_panel if val_panel is not None else auto_val
        test_p = test_panel if test_panel is not None else auto_test
    else:
        train_p, val_p, test_p = panel, val_panel, test_panel

    # 2) 对每个 seed 迭代搜索 → 收集 best 表达式
    import asyncio

    def _make_searcher():
        kw = ({"rounds": config.rounds} if config.algo == "co"
              else {"generations": config.rounds} if config.algo == "ea"
              else {"depth": config.rounds})
        return create_algo(config.algo, provider=provider, **kw)

    candidates: Dict[str, str] = {}   # expression -> seed
    for seed in config.seeds:
        if not seed or not seed.strip():
            continue
        try:
            searcher = _make_searcher()
            result = asyncio.run(searcher.run(
                seed, train_p, forward_periods=config.forward_periods,
                market=config.market, knowledge_context=config.knowledge_context))
            best = result.best_expression
            if best and best not in candidates:
                candidates[best] = seed
        except Exception as exc:  # noqa: BLE001
            _logger.warning("seed %s 搜索失败: %s", seed, exc)

    if not candidates:
        # 搜索无产出：退回直接评估 seed
        for s in config.seeds:
            if s and s.strip():
                candidates.setdefault(s, s)

    # 3) 去冗余（train 期）→ 代表因子
    reps = dedup_expressions(
        list(candidates.keys()), train_p,
        correlation_threshold=config.dedup_threshold,
        min_abs_metric=config.min_abs_ic,
        forward_periods=config.forward_periods, market=config.market,
        compute_ic=True,
    )
    rep_exprs = [r["name"] for r in reps]

    # 4) 可选 judge 重排序（真实 LLM 打分）
    if config.run_judge and rep_exprs:
        scores = _run_judge(rep_exprs, provider)
        rep_exprs = sorted(rep_exprs, key=lambda e: -scores.get(e, 0.0))

    # 5) 逐代表：train IC + val IC + test OOS 回测
    steps: List[Dict[str, object]] = []
    for expr in rep_exprs[: config.max_candidates]:
        tr, val = _eval_ic(expr, train_p, config.forward_periods, config.market,
                           val_panel=val_p)
        step = StepReport(expression=expr, algo=config.algo,
                          seed=candidates.get(expr, ""),
                          train_ic=tr.get("ic", float("nan")),
                          train_rank_ic=tr.get("rank_ic", float("nan")),
                          val_ic=val.get("ic"))
        # 该代表在去重中吸收的冗余因子
        for r in reps:
            if r["name"] == expr:
                step.removed_redundant = [m for m in r["cluster"] if m != expr]
                break
        # test OOS 回测
        if test_p is not None and len(test_p.dates) >= config.n_groups:
            try:
                bt = factor_expression_backtest(
                    expr, test_p, forward_periods=config.forward_periods,
                    n_groups=config.n_groups, long_short=config.long_short,
                    cost_rate=config.cost_rate)
                pf = bt["portfolio"]
                step.test_ic = (bt["ic_report"] or {}).get("ic_mean")
                step.test_sharpe = pf.get("sharpe_annual")
                step.test_return = pf.get("total_return")
                step.test_mdd = pf.get("max_drawdown")
                step.test_net_sharpe = pf.get("net_sharpe_annual")
                step.turnover_mean = pf.get("turnover_mean")
                step.cost_ratio = pf.get("cost_ratio")
                # 宽松闸门：换手超上限或净成本吃掉过度 → 标记（不移除展示）
                if (config.max_turnover and config.max_turnover > 0
                        and step.turnover_mean is not None
                        and step.turnover_mean > config.max_turnover):
                    step.high_turnover = True
                step.daily_returns = [float(x) for x in pf.get("daily_returns") or []]
                # 逐日截面 IC（供前端画 IC 时序，观察稳定性）
                try:
                    step.ic_series = _daily_ic_series(expr, test_p, config.forward_periods)
                except Exception as exc:  # noqa: BLE001
                    _logger.debug("IC 时序计算失败 %s: %s", expr, exc)
                    step.ic_series = []
            except Exception as exc:  # noqa: BLE001
                _logger.debug("test OOS 回测失败 %s: %s", expr, exc)
        elif test_p is not None:
            # panel 太小无法分组，至少给个 test IC
            _, test_metrics = _eval_ic(expr, test_p, config.forward_periods,
                                       config.market, val_panel=None)
            step.test_ic = test_metrics.get("ic")
        steps.append(step.to_dict())

    # 6)（可选）持久化 (expr, IC) 配对 -> 未来 SFT / 弱标签
    if config.persist_pairs and steps:
        try:
            from .factors.seed_pool import FactorPairStore
            store = FactorPairStore()
            pairs = [
                (s["expression"], s["train_ic"], s["train_rank_ic"])
                for s in steps if s["train_ic"] is not None
            ]
            if pairs:
                store.add_pairs(pairs, market=config.market,
                                forward_periods=config.forward_periods)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("持久化因子配对失败: %s", exc)

    # 7)（可选）复合组合：在 test 期把代表因子合成一个可交易 alpha 组合
    composite_res: Optional[Dict[str, object]] = None
    if config.net_gate:
        # 宽松净成本/换手闸门：把高换手或净成本吃光的因子从复合池剔除（仍留在展示）
        has_turn_gate = config.max_turnover and config.max_turnover > 0
        has_net_gate = config.min_net_sharpe and config.min_net_sharpe > 0
        if has_turn_gate or has_net_gate:
            _step_by_expr = {s["expression"]: s for s in steps}
            filtered = []
            for expr in rep_exprs:
                st = _step_by_expr.get(expr, {})
                if st.get("high_turnover"):
                    continue
                ns = st.get("test_net_sharpe")
                if has_net_gate and ns is not None and ns < config.min_net_sharpe:
                    continue
                filtered.append(expr)
            _logger.info("净成本闸门：%d→%d 个代表因子进入复合", len(rep_exprs), len(filtered))
            rep_exprs = filtered
    if (config.run_composite and rep_exprs and test_p is not None
            and len(test_p.dates) >= config.n_groups):
        try:
            composite_res = composite_backtest(
                rep_exprs, test_p,
                training_panel=train_p,          # 权重在 train 上拟合，test 上 OOS
                scheme=config.composite_scheme,
                forward_periods=config.forward_periods,
                n_groups=config.n_groups,
                long_short=config.long_short,
                cost_rate=config.cost_rate,
                standardize=config.composite_standardize,
                market=config.market,
            )
            composite_res.pop("composite", None)  # 避免返回整个大面板
            _attach_composite_contribution(composite_res, steps)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("复合组合回测失败: %s", exc)

    # 汇总
    n_test = len(steps)
    def _mean(key):
        vals = [s[key] for s in steps if s.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    summary = {
        "seed_count": len(config.seeds),
        "candidate_count": len(candidates),
        "representative_count": len(rep_exprs),
        "backtested_count": len(steps),
        "mean_train_ic": _mean("train_ic"),
        "mean_val_ic": _mean("val_ic"),
        "mean_test_ic": _mean("test_ic"),
        "mean_test_sharpe": _mean("test_sharpe"),
        "algo": config.algo,
        "rounds": config.rounds,
    }
    return {
        "config": {
            "seeds": config.seeds, "algo": config.algo, "rounds": config.rounds,
            "dedup_threshold": config.dedup_threshold,
            "run_composite": config.run_composite,
            "composite_scheme": config.composite_scheme,
            "cast": (n_test, len(train_p.dates), len(val_p.dates) if val_p else 0,
                     len(test_p.dates) if test_p else 0),
        },
        "steps": steps,
        "composite": composite_res,
        "summary": summary,
    }
