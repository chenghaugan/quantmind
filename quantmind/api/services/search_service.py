"""SearchService: 表达式截面评估 + 因子迭代搜索（CoT）的 API 服务层。

复用 DataManager 构造多标的面板（index=日期, columns=标的），把 P0 的
``evaluate_expression`` 与 P1 的 ``FactorSearcher.cot_search`` 暴露为
可供 FastAPI / CLI 调用的方法。

与现状一致：无 LLM key 或离线时，CoT 回落到确定性变异器，保证流程可跑通。
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...core.constant import Exchange, Interval
from ...data import DataManager
from ...data.feed.base import HistoryRequest
from ...research import (
    evaluate_expression as _eval_expr,
    batch_evaluate_expressions,
    FactorSearcher,
    SearchResult,
    create_algo,
    list_algos,
    dedup_expressions as _dedup_exprs,
    factor_expression_backtest as _expr_backtest,
    run_pipeline as _run_pipeline,
    PipelineConfig,
    E2EConfig,
    run_e2e as _run_e2e,
)
from ...knowledge import KnowledgeStore
from ...research.factors.alpha_cs import Panel
from ..schemas import FactorE2ERequest

_logger = logging.getLogger("quantmind.api")

#: e2e 短期结果缓存 TTL（秒）。避免前端重复发起 85s+ 长跑。
_E2E_CACHE_TTL = 20 * 60


def _sanitize(o: Any) -> Any:
    """把 numpy float/NaN/datetime 规整为 JSON 可序列化值。"""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, (int, str, bool)) or o is None:
        return o
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(x) for x in o]
    if isinstance(o, datetime):
        return o.isoformat()
    return o


def _flt(x) -> Optional[float]:
    """把可能是 NaN 的 float 规整为 float 或 None（用于 metric 展示）。"""
    try:
        f = float(x)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _split_vt_symbol(symbol: str, default_exchange: str):
    """把标的拆成 (symbol, exchange)。支持 ``rb0.SHFE`` 形式（跨交易所统一面板）。

    含 ``.`` 时按最后一段解析为该标的自有交易所；否则用全局默认交易所。
    """
    sym = symbol.strip()
    if "." in sym:
        head, _, exch = sym.rpartition(".")
        if head and exch:
            return head.strip(), exch.strip().upper()
    return sym, default_exchange.upper()


class SearchService:
    """因子表达式评估与迭代搜索服务。"""

    def __init__(self, dm: DataManager, provider=None) -> None:
        self.dm = dm
        self.provider = provider  # 可选 LLMProvider；None → CoT 回落 mock 变异器
        # e2e 短期结果缓存：key -> (写入时刻, 结果 dict)
        self._e2e_cache: Dict[str, tuple] = {}

    # -- 面板构造（复用 CrossSectionService 逻辑） ---------------------------
    async def _build_panel(
        self,
        symbols: List[str],
        exchange: str = "SHFE",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        source_sink: Optional[Dict[str, str]] = None,
    ) -> Panel:
        symbols = [s for s in (symbols or []) if s and s.strip()]
        if len(symbols) < 2:
            raise ValueError("表达式截面研究至少需要 2 个标的")
        default_exch = exchange.upper()
        interv = Interval(interval or "1d")
        tasks = []
        for s in symbols:
            sym, exch_str = _split_vt_symbol(s, default_exch)
            tasks.append(self.dm.get_bar_data(
                HistoryRequest(
                    symbol=sym,
                    exchange=Exchange(exch_str),
                    interval=interv,
                    start=datetime.fromisoformat(start) if start else None,
                    end=datetime.fromisoformat(end) if end else None,
                ),
                source_sink=source_sink,
            ))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        bars_by_symbol: Dict[str, list] = {}
        missing: List[str] = []
        for sym, res in zip(symbols, results):
            if isinstance(res, Exception) or not res:
                missing.append(sym)
                continue
            bars_by_symbol[sym] = res
        if len(bars_by_symbol) < 2:
            raise ValueError(
                f"可用标的不足 2 个（缺失: {missing or '无'}）："
                "请确认行情数据可用"
            )
        return Panel.from_bars(bars_by_symbol)

    # -- 表达式评估 ----------------------------------------------------------
    async def evaluate_expression(
        self,
        expression: str,
        symbols: List[str],
        exchange: str = "SHFE",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        forward_periods: int = 1,
        market: str = "",
    ) -> dict:
        """对单个表达式做「面板求值 → 截面 IC 评估」，返回报告 dict。"""
        panel = await self._build_panel(symbols, exchange, interval, start, end)
        loop = asyncio.get_running_loop()
        rep = await loop.run_in_executor(
            None,
            lambda: _eval_expr(expression, panel, forward_periods=forward_periods,
                               market=market, use_cache=False),
        )
        out = rep.to_dict()
        out["n_symbols"] = len(panel.symbols)
        out["n_dates"] = len(panel.dates)
        out["symbols"] = list(panel.symbols)
        dates = list(panel.dates)
        out["date_range"] = [dates[0].isoformat() if dates else None,
                             dates[-1].isoformat() if dates else None]
        return _sanitize(out)

    async def evaluate_expressions_batch(
        self,
        expressions: List[str],
        symbols: List[str],
        exchange: str = "SHFE",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        forward_periods: int = 1,
        market: str = "",
    ) -> List[dict]:
        """批量评估多个表达式。"""
        panel = await self._build_panel(symbols, exchange, interval, start, end)
        loop = asyncio.get_running_loop()
        reports = await loop.run_in_executor(
            None,
            lambda: batch_evaluate_expressions(
                expressions, panel, forward_periods=forward_periods,
                market=market, use_cache=False,
            ),
        )
        return [_sanitize(r.to_dict()) for r in reports]

    # -- 迭代搜索（co / ea / tot） ---------------------------------------------
    async def search(
        self,
        seed: str,
        symbols: List[str],
        exchange: str = "SHFE",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        algo: str = "co",
        rounds: int = 6,
        forward_periods: int = 1,
        market: str = "",
        val_symbols: Optional[List[str]] = None,
        val_start: Optional[str] = None,
        val_end: Optional[str] = None,
    ) -> dict:
        """对 seed 表达式做指定算法（co/ea/tot）的迭代搜索，返回 ``SearchResult`` dict。

        可选 ``val_symbols/val_start/val_end`` 提供独立验证期面板做防泄漏评估。
        ``algo`` 未知时按默认回落 ``co``（链式精炼）。
        """
        panel = await self._build_panel(symbols, exchange, interval, start, end)

        val_panel: Optional[Panel] = None
        if val_symbols:
            val_panel = await self._build_panel(
                val_symbols, exchange, interval, val_start, val_end)

        algo_name = algo if algo in list_algos() else "co"
        # 每类算法的迭代参数：co=rounds, ea=generations, tot=depth
        algo_kwargs = {
            "co": {"rounds": rounds},
            "ea": {"generations": rounds},
            "tot": {"depth": rounds},
        }.get(algo_name, {"rounds": rounds})

        searcher = create_algo(algo_name, provider=self.provider, **algo_kwargs)

        async def _run() -> SearchResult:
            return await searcher.run(
                seed, panel, val_panel=val_panel,
                forward_periods=forward_periods, market=market,
            )

        # search 内部是 async（LLM/评估），直接在事件循环中跑（评估经 executor）
        result = await _run()
        out = result.to_dict()
        out["algo"] = algo_name
        out["n_symbols"] = len(panel.symbols)
        out["date_range"] = [panel.dates[0].isoformat() if len(panel.dates) else None,
                             panel.dates[-1].isoformat() if len(panel.dates) else None]
        return _sanitize(out)

    async def cot_search(
        self,
        seed: str,
        symbols: List[str],
        exchange: str = "SHFE",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        rounds: int = 6,
        forward_periods: int = 1,
        market: str = "",
        val_symbols: Optional[List[str]] = None,
        val_start: Optional[str] = None,
        val_end: Optional[str] = None,
        algo: str = "co",
    ) -> dict:
        """``cot_search`` 为 :meth:`search` 的向后兼容别名（默认 algo=co）。"""
        return await self.search(
            seed, symbols, exchange=exchange, interval=interval, start=start,
            end=end, algo=algo, rounds=rounds, forward_periods=forward_periods,
            market=market, val_symbols=val_symbols, val_start=val_start, val_end=val_end,
        )

    # -- 因子去冗余（相关性聚类） --------------------------------------------
    async def dedup(
        self,
        expressions: List[str],
        symbols: List[str],
        exchange: str = "SHFE",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        correlation_threshold: float = 0.7,
        min_abs_metric: float = 0.0,
        forward_periods: int = 1,
        market: str = "",
        compute_ic: bool = True,
    ) -> dict:
        """对一批表达式做相关性聚类去冗余，返回每簇代表性因子。"""
        panel = await self._build_panel(symbols, exchange, interval, start, end)
        loop = asyncio.get_running_loop()
        kept = await loop.run_in_executor(
            None,
            lambda: _dedup_exprs(
                expressions, panel, correlation_threshold=correlation_threshold,
                min_abs_metric=min_abs_metric, forward_periods=forward_periods,
                market=market, compute_ic=compute_ic,
            ),
        )
        return {
            "n_input": len([e for e in expressions if e and e.strip()]),
            "n_input_unique": len(expressions),
            "n_kept": len(kept),
            "representatives": [
                {"expression": c["name"], "metric": _flt(c["metric"]),
                 "n_removed": len(c["cluster"]) - 1,
                 "cluster": list(c["cluster"])}
                for c in kept
            ],
            "correlation_threshold": correlation_threshold,
        }

    # -- 表达式 → 截面多空组合回测（研究闭环） ------------------------------
    async def backtest_expression(
        self,
        expression: str,
        symbols: List[str],
        exchange: str = "SHFE",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        forward_periods: int = 1,
        n_groups: int = 5,
        long_short: bool = True,
        cost_rate: float = 0.0,
    ) -> dict:
        """对挖掘出的 DSL 因子表达式直接做截面多空组合回测。"""
        panel = await self._build_panel(symbols, exchange, interval, start, end)
        loop = asyncio.get_running_loop()
        out = await loop.run_in_executor(
            None,
            lambda: _expr_backtest(
                expression, panel, forward_periods=forward_periods,
                n_groups=n_groups, long_short=long_short, cost_rate=cost_rate,
            ),
        )
        return _sanitize(out)

    # -- 端到端因子挖掘流水线（挖掘→去冗余→逐因子OOS→复合组合） --------------
    async def pipeline(
        self,
        seeds: List[str],
        symbols: List[str],
        exchange: str = "SHFE",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        algo: str = "co",
        rounds: int = 3,
        forward_periods: int = 1,
        market: str = "",
        dedup_threshold: float = 0.7,
        min_abs_ic: float = 0.0,
        train_frac: float = 0.6,
        val_frac: float = 0.2,
        run_composite: bool = True,
        composite_scheme: str = "icir",
        n_groups: int = 5,
        long_short: bool = True,
        cost_rate: float = 0.0,
        max_candidates: int = 8,
    ) -> dict:
        """端到端因子挖掘流水线。

        在标的面板上：每个 seed 用指定算法（co/ea/tot）迭代挖掘 → 相关性去冗余
        → 防泄漏切分（train/val/test）→ 逐代表做 test 期 OOS 多空回测 →（可选）
        用组合权重方案把代表合成为复合 alpha 并回测。
        """
        sources: Dict[str, str] = {}
        panel = await self._build_panel(symbols, exchange, interval, start, end,
                                        source_sink=sources)
        seed_list = [s for s in (seeds or []) if s and s.strip()]
        if not seed_list:
            raise ValueError("至少需要 1 个 seed 表达式")
        cfg = PipelineConfig(
            seeds=seed_list,
            algo=algo if algo in ("co", "ea", "tot") else "co",
            rounds=rounds,
            forward_periods=forward_periods,
            train_frac=train_frac,
            val_frac=val_frac,
            market=market,
            dedup_threshold=dedup_threshold,
            min_abs_ic=min_abs_ic,
            run_composite=run_composite,
            composite_scheme=composite_scheme,
            n_groups=n_groups,
            long_short=long_short,
            cost_rate=cost_rate,
            max_candidates=max_candidates,
            persist_pairs=False,
        )
        loop = asyncio.get_running_loop()
        report = await loop.run_in_executor(
            None, lambda: _run_pipeline(panel, config=cfg, provider=self.provider),
        )

        # 去掉不可 JSON 序列化的内存对象（复合面板 DataFrame）
        composite = report.get("composite")
        if isinstance(composite, dict):
            composite.pop("composite", None)

        # 数据源透明度：到底用了真实行情还是 mock
        src_names = {k: v for k, v in sources.items() if v}

        # 本地行情仓库（Parquet 写缓存）状态：用于提示「是否已用秒级缓存而非联网拉取」
        cache_info: Dict[str, Any] = {"enabled": bool(self.dm.disk_cache)}
        if self.dm.disk_cache is not None:
            try:
                cache_info.update(self.dm.disk_cache.stats())
            except Exception:  # noqa: BLE001
                pass

        out = {
            "algo": report["config"]["algo"],
            "n_symbols": len(panel.symbols),
            "n_dates": len(panel.dates),
            "date_range": [panel.dates[0].isoformat() if len(panel.dates) else None,
                           panel.dates[-1].isoformat() if len(panel.dates) else None],
            "data_sources": src_names,
            "is_real": bool(src_names) and any(v and v != "mock" for v in src_names.values()),
            "cache": cache_info,
            "summary": _sanitize(report["summary"]),
            "steps": _sanitize(report["steps"]),
            "composite": _sanitize(composite),
        }
        return out

    # -- 端到端编排（AI 证据 → 挖掘 → 复合 → 策略代码）+ 可选沉淀知识库 ----------
    async def e2e(self, req: FactorE2ERequest, ingest: bool = True) -> dict:
        """端到端因子研究。

        在标的面板上跑 :func:`quantmind.research.run_e2e`（AI 证据研究 →
        因子挖掘 → OOS 复合 alpha → 策略代码），返回统一契约 dict。

        当 ``ingest`` 为真时，把 report 产出的每条因子 + 策略 + 研究过程日志
        沉淀进 :class:`KnowledgeStore`，返回值附加 ``knowledge`` 字段
        （``{"ingested": True, "kb_records": n}``）。
        strategy code 保留完整不截断。
        """
        # 短期结果缓存：同一 (idea, 标的集, 交易所, 算法, 轮数, 前向期, ingest) 在 TTL 内
        # 复用上次结果，避免前端重复点击触发 85s+ 长跑；参数/标的/idea 变化即换新 key。
        cache_key = (str(req.idea),
                     tuple(sorted(s for s in (req.symbols or []) if s)),
                     str(req.exchange), str(req.algo), int(req.rounds),
                     int(req.forward_periods), bool(ingest))
        cached = self._e2e_cache.get(cache_key)
        if cached is not None and (time.time() - cached[0]) < _E2E_CACHE_TTL:
            out = dict(cached[1])
            out["cached"] = True
            return out

        panel = await self._build_panel(req.symbols, req.exchange, req.interval,
                                        req.start, req.end)
        symbols = [s for s in (req.symbols or []) if s and s.strip()]

        # 知识库反哺（持续学习·消费端）：用 idea 检索历史已验证因子，
        # 把命中表达式作为额外种子注入挖掘，让相似想法复用已验证的 alpha 起点。
        seeds = self._kb_context_seeds(req.idea, user_seeds=req.seeds or [])

        cfg = E2EConfig(
            idea=req.idea,
            asset_class=req.asset_class,
            seeds=seeds,
            algo=req.algo if req.algo in ("co", "ea", "tot") else "co",
            rounds=req.rounds,
            forward_periods=req.forward_periods,
            market=req.market,
            train_frac=req.train_frac,
            val_frac=req.val_frac,
            dedup_threshold=req.dedup_threshold,
            min_abs_ic=req.min_abs_ic,
            run_composite=req.run_composite,
            composite_scheme=req.composite_scheme,
            composite_standardize=req.composite_standardize,
            n_groups=req.n_groups,
            long_short=req.long_short,
            cost_rate=req.cost_rate,
            max_candidates=req.max_candidates,
            verify_threshold=req.verify_threshold,
            run_search=req.run_search,
            max_rounds=req.max_rounds,
        )

        loop = asyncio.get_running_loop()
        report = await loop.run_in_executor(
            None, lambda: _run_e2e(panel, config=cfg, provider=self.provider),
        )

        out = _sanitize(report)
        out["n_symbols"] = len(panel.symbols)
        out["date_range"] = [panel.dates[0].isoformat() if len(panel.dates) else None,
                             panel.dates[-1].isoformat() if len(panel.dates) else None]

        # 构建溯源信息（对标 Vibe-Trading evidence chain）
        evidence = report.get("evidence") or {}
        out["provenance"] = {
            "data_sources": [f"{req.exchange}:{s}" for s in (req.symbols or []) if s],
            "tool_calls": [
                {"tool": "run_e2e", "config": {
                    "algo": cfg.algo, "rounds": cfg.rounds,
                    "forward_periods": cfg.forward_periods,
                }},
            ],
            "evidence_chain": evidence.get("hypotheses") or [],
            "verified_exprs": evidence.get("verified_exprs") or [],
            "generated_at": report.get("generated_at"),
        }

        if ingest:
            n = self._ingest_report(report, idea=req.idea, symbols=symbols,
                                    asset_class=req.asset_class, market=req.market)
            out["knowledge"] = {"ingested": True, "kb_records": n}

        return out

    def _ingest_report(self, report: dict, idea: str, symbols: List[str],
                       asset_class: str = "", market: str = "") -> int:
        """把 e2e report 沉淀进知识库，返回写入的记录条数。

        - 每个 evidence 因子 → ``ingest_factor``
        - strategy → ``ingest_strategy``
        - 研究过程假设 → ``ingest_research_log``
        """
        store = KnowledgeStore()
        n = 0
        try:
            evidence = report.get("evidence") or {}
            hypotheses = evidence.get("hypotheses") or []
            factors = evidence.get("factors") or []
            verified = set(evidence.get("verified_exprs") or [])

            for f in factors:
                name = (f or {}).get("name") or ""
                expr = (f or {}).get("expression") or ""
                if not name and not expr:
                    continue
                status = "verified" if (expr and expr in verified) else "active"
                store.ingest_factor(
                    name=name, expression=expr, idea=idea,
                    ic=None, ir=None, status=status,
                    symbols=symbols, asset_class=asset_class, market=market,
                )
                n += 1

            strategy = report.get("strategy") or {}
            if strategy.get("code"):
                composite = report.get("pipeline", {}).get("composite") or {}
                scheme = composite.get("scheme") or (report.get("pipeline", {}).get("config", {})
                                                     .get("composite_scheme") or "")
                sharpe = _flt(composite.get("sharpe"))
                store.ingest_strategy(
                    code=str(strategy.get("code", "")),
                    code_safe=bool(strategy.get("code_safe")),
                    idea=idea,
                    composite_scheme=str(scheme or ""),
                    composite_sharpe=sharpe,
                    symbols=symbols,
                )
                n += 1

            store.ingest_research_log(
                idea=idea,
                hypotheses=hypotheses,
                evidence=dict(evidence),
            )
            n += 1
        except Exception as exc:  # noqa: BLE001
            _logger.warning("知识库沉淀失败（不影响主流程返回）: %s", exc)
        return n

    @staticmethod
    def _kb_context_seeds(idea: str, user_seeds: List[str], top_k: int = 6) -> List[str]:
        """知识库检索反哺：用 idea 检索历史已验证因子表达式，作挖掘种子补充。

        持续学习闭环的「消费」端——之前 e2e 沉淀进知识库的 VERIFIED 因子表达式，
        在相似的新想法下被检索出来，与用户显式种子合并去重，作为因子挖掘起点。

        知识库为空 / 检索失败时静默返回用户种子（不阻断主流程）。
        """
        user = [s for s in (user_seeds or []) if s and s.strip()]
        added: List[str] = []
        try:
            hits = KnowledgeStore().search(idea, top_k=top_k, kind="factor")
            for h in hits or []:
                expr = (h.get("metadata") or {}).get("expression") or ""
                if expr and expr.strip() and expr not in user and expr not in added:
                    added.append(expr.strip())
        except Exception as exc:  # noqa: BLE001
            _logger.debug("知识库反哺检索失败（沿用用户种子）: %s", exc)
        if not added:
            return user
        return list(dict.fromkeys([*added, *user]))
