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
import uuid
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


def _num(x) -> Optional[float]:
    """安全取数值（NaN/None/非法 → None），用于门槛判定指标提取。"""
    if x is None:
        return None
    try:
        f = float(x)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _hash_gate(gate) -> str:
    """把 gate 配置压成可哈希字符串（用于 e2e 缓存 key）。"""
    if not gate:
        return ""
    return "|".join(f"{k}={gate.get(k)}" for k in sorted(gate))


def _e2e_strategy_id(idea: str, symbols: list) -> str:
    """为一次端到端挖掘生成稳定、可读的 strategy_id（同日同 idea 复用同 id）。"""
    import hashlib
    import time as _time

    h = hashlib.sha1(
        f"{idea}|{sorted(s for s in (symbols or []) if s)}".encode("utf-8")
    ).hexdigest()[:8]
    return f"e2e_{h}_{_time.strftime('%Y%m%d')}"


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
        min_abs_ic: float = 0.03,
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
        # 短期结果缓存：同一 (idea, 标的集, 交易所, 算法, 轮数, 前向期, ingest, gate, promote) 在 TTL 内
        # 复用上次结果，避免前端重复点击触发 85s+ 长跑；参数/标的/idea 变化即换新 key。
        cache_key = (str(req.idea),
                     tuple(sorted(s for s in (req.symbols or []) if s)),
                     str(req.exchange), str(req.interval), str(req.algo), int(req.rounds),
                     int(req.forward_periods), bool(ingest),
                     _hash_gate(getattr(req, "gate", None)),
                     bool(getattr(req, "promote", False)))
        cached = self._e2e_cache.get(cache_key)
        if cached is not None and (time.time() - cached[0]) < _E2E_CACHE_TTL:
            out = dict(cached[1])
            out["cached"] = True
            return out

        panel = await self._build_panel(req.symbols, req.exchange, req.interval,
                                        req.start, req.end)
        symbols = [s for s in (req.symbols or []) if s and s.strip()]

        # 领域知识获取层·开箱即用：确保内置方法论种子（缠论/威科夫/海龟等）已入库，
        # 供 enrich_idea 检索 + Web 知识库页浏览（幂等，按 title 去重）。
        try:
            from ...knowledge.seeds import ensure_seed_data
            ensure_seed_data()
        except Exception:  # noqa: BLE001 —— 种子缺失不影响主流程
            pass

        # 用户补充的方法论（needs_input 澄清后带上的信息）：先作为新方法论入库，
        # 使本次 e2e 的 enrich_idea 能命中，也让下次同类想法直接可用（持续学习·写回）。
        meth_input = getattr(req, "methodology_input", None)
        if meth_input and str(meth_input).strip():
            try:
                from ...knowledge import KnowledgeStore
                KnowledgeStore().ingest_methodology(
                    title=f"{req.idea}（用户补充）",
                    concept=str(req.idea),
                    content=str(meth_input).strip(),
                    source="user",
                    tags=["方法论", "用户补充"],
                    meta={"implementable": False, "evidence": "user"},
                )
            except Exception as exc:  # noqa: BLE001
                _logger.debug("用户方法论补充入库失败（不影响主流程）: %s", exc)

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
            use_knowledge=getattr(req, "use_knowledge", True),
            web_fallback=getattr(req, "web_fallback", True),
        )

        loop = asyncio.get_running_loop()

        # 持续学习闭环·实时参考：注入历史成功因子模式 + 失败避坑 到挖掘 LLM prompt。
        # 空库/失败时不注入，搜索保持原行为。
        knowledge_context: Dict[str, Any] = {}
        try:
            from ...research.knowledge_loop import kb_search_context
            knowledge_context = await kb_search_context(
                KnowledgeStore(), idea=req.idea, max_success=8, max_fail=8)
        except Exception as exc:  # noqa: BLE001
            _logger.debug("知识库上下文检索失败，本次不注入（不影响主流程）: %s", exc)
            knowledge_context = {}

        report = await loop.run_in_executor(
            None, lambda: _run_e2e(panel, config=cfg, provider=self.provider,
                                   knowledge_context=knowledge_context),
        )

        out = _sanitize(report)
        if out.get("needs_input"):
            # 方法论知识层判定无法忠实实现：不回传伪结果，提示用户补充信息。
            return out
        out["n_symbols"] = len(panel.symbols)
        out["date_range"] = [panel.dates[0].isoformat() if len(panel.dates) else None,
                             panel.dates[-1].isoformat() if len(panel.dates) else None]

        # 构建溯源信息（对标 Vibe-Trading evidence chain）
        evidence = report.get("evidence") or {}
        knowledge = report.get("knowledge") or {}
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
            "knowledge_sources": list(knowledge.get("sources") or []),
            "generated_at": report.get("generated_at"),
        }

        # 领域知识摘要（若有）并入统一契约
        if knowledge:
            out["knowledge"] = {
                "concept": knowledge.get("concept", ""),
                "definition": knowledge.get("definition", ""),
                "buy_signal_rules": list(knowledge.get("buy_signal_rules") or []),
                "candidate_factors": list(knowledge.get("candidate_factors") or []),
                "sources": list(knowledge.get("sources") or []),
                "kb_hits": list(knowledge.get("kb_hits") or []),
            }

        if ingest:
            ingested = await self._ingest_report(
                report, idea=req.idea, symbols=symbols,
                asset_class=req.asset_class, market=req.market)
            out["knowledge"] = out.get("knowledge") or {}
            out["knowledge"]["ingested"] = True
            out["knowledge"]["kb_records"] = int(ingested.get("kb_records", 0))
            for k in ("run_id", "judged_trials", "brief",
                      "effective_themes", "failure_traps", "next_suggestions"):
                if ingested.get(k):
                    out["knowledge"][k] = ingested[k]

        # 门槛判定 + 达标自动入有效策略库（新增，默认关闭，向后兼容）
        # 仅当请求携带 ``gate`` 配置时启用；判定/入库全程失败闭合，
        # 任何异常只记录到返回 dict 的 ``gate`` 字段，绝不影响主报告。
        if getattr(req, "gate", None):
            out["gate"] = await self._gate_judge_and_promote(report, req)

        return out

    async def _ingest_report(self, report: dict, idea: str, symbols: List[str],
                             asset_class: str = "", market: str = "",
                             store: Optional[KnowledgeStore] = None) -> dict:
        """把 e2e report 沉淀进知识库，返回写入摘要 dict。

        - 每个 evidence 因子 → ``ingest_factor``
        - strategy → ``ingest_strategy``
        - 研究过程假设 → ``ingest_research_log``
        - 领域知识摘要（KnowledgeBrief）→ ``ingest_methodology``（title=concept，幂等）

        新增「AI 持续学习闭环」落库（C1）：
        - 用 :func:`quantmind.research.knowledge_loop.run_knowledge_loop` 判读每个
          代表因子（verified/active/rejected + reason + tags）并生成经验 brief；
        - ``start_e2e_run`` 建 run 行 → 每个 judged trial 落 ``factor_trials`` →
          ``finish_e2e_run`` 回填统计与 brief。

        返回：``{"kb_records": n, "run_id", "judged_trials", "brief",
        "effective_themes", "failure_traps", "next_suggestions"}``。
        """
        kb = store or KnowledgeStore()
        n = 0
        try:
            # 领域知识摘要：把 brief 作为方法论条目落库（title=concept，幂等）
            knowledge = report.get("knowledge") or {}
            concept = str(knowledge.get("concept") or "").strip()
            if concept:
                guidance = "\n".join(
                    "- " + str(r) for r in (knowledge.get("buy_signal_rules") or [])
                )
                summary = (knowledge.get("definition")
                           or "由领域知识增强层提炼的方法论摘要。" or "")
                kb.ingest_methodology(
                    title=concept,
                    concept=concept,
                    summary=str(summary),
                    content=guidance,
                    source="knowledge_enrichment",
                    tags=[f"{c.get('kind')}" for c in
                          (knowledge.get("candidate_factors") or []) if isinstance(c, dict)],
                )
                n += 1

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
                kb.ingest_factor(
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
                kb.ingest_strategy(
                    code=str(strategy.get("code", "")),
                    code_safe=bool(strategy.get("code_safe")),
                    idea=idea,
                    composite_scheme=str(scheme or ""),
                    composite_sharpe=sharpe,
                    symbols=symbols,
                )
                n += 1

            kb.ingest_research_log(
                idea=idea,
                hypotheses=hypotheses,
                evidence=dict(evidence),
            )
            n += 1
        except Exception as exc:  # noqa: BLE001
            _logger.warning("知识库沉淀失败（不影响主流程返回）: %s", exc)

        # ---------- AI 持续学习闭环落库（C1，try/except 隔离不阻断主返回） ----------
        loop_out: Dict[str, Any] = {
            "kb_records": n,
            "run_id": "",
            "judged_trials": [],
            "brief": "",
            "effective_themes": [],
            "failure_traps": [],
            "next_suggestions": [],
        }
        try:
            from ...research.knowledge_loop import run_knowledge_loop

            loop_res = await run_knowledge_loop(kb, self.provider, report, idea=idea)
            # 建 run 行：run_id 用 run-<ts>-<hex6>，保证唯一且含时间信息
            run_id = (f"run-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                      f"-{uuid.uuid4().hex[:6]}")
            pipe = report.get("pipeline", {}) or {}
            pcfg = pipe.get("config", {}) or {}
            algo = str(pcfg.get("algo") or "")
            rounds = int(pcfg.get("rounds") or 0)
            scheme = str((pipe.get("composite") or {}).get("scheme") or "")
            kb.start_e2e_run(
                run_id=run_id, idea=idea, asset_class=asset_class, market=market,
                symbols=symbols, exchange="", interval="", algo=algo, rounds=rounds,
                forward_periods=int(pcfg.get("forward_periods") or 0),
                status="running",
            )

            # 代表因子集合：去冗余后各代表（含 removed_redundant）作为 is_representative
            rep_exprs = {s.get("expression") for s in (pipe.get("steps") or [])
                         if isinstance(s, dict) and s.get("expression")}
            summary = report.get("summary") or {}
            n_rep = int(summary.get("representative_count") or 0) or len(rep_exprs)

            judged_trials: List[Dict[str, Any]] = []
            for t in loop_res.get("trials") or []:
                expression = t.get("expression") or ""
                if not expression:
                    continue
                is_rep = bool(expression in rep_exprs)
                try:
                    kb.ingest_factor_trial(
                        run_id=run_id,
                        expression=expression,
                        algo=algo or str(t.get("algo") or ""),
                        seed=str(t.get("seed") or ""),
                        train_ic=_flt(t.get("train_ic")),
                        val_ic=_flt(t.get("val_ic")),
                        test_ic=_flt(t.get("test_ic")),
                        test_sharpe=_flt(t.get("test_sharpe")),
                        test_return=_flt(t.get("test_return")),
                        test_mdd=_flt(t.get("test_mdd")),
                        is_representative=is_rep,
                        status=str(t.get("status") or "active"),
                        reason=str(t.get("reason") or ""),
                        removed_redundant=(
                            [str(x) for x in t["removed_redundant"]]
                            if isinstance(t.get("removed_redundant"), list) else None
                        ),
                    )
                except Exception as exc:  # noqa: BLE001
                    _logger.warning("因子试验落库失败 %s: %s", expression, exc)
                judged_trials.append({
                    "expression": expression,
                    "status": str(t.get("status") or "active"),
                    "reason": str(t.get("reason") or ""),
                    "tags": [str(x) for x in (t.get("tags") or [])],
                })
                n += 1

            # 回填统计 + AI brief
            composite = pipe.get("composite") or {}
            n_verified = int(summary.get("n_verified_hypotheses")
                             if summary.get("n_verified_hypotheses") is not None
                             else len(evidence.get("verified_exprs") or []))
            kb.finish_e2e_run(
                run_id=run_id,
                n_representative=n_rep,
                n_verified_hypotheses=n_verified,
                composite_scheme=str(scheme or ""),
                composite_fwd_ic=_flt(composite.get("ic_mean")),
                composite_sharpe=_flt(composite.get("sharpe")),
                brief=str(loop_res.get("brief") or ""),
                status="done",
            )
            loop_out.update({
                "run_id": run_id,
                "judged_trials": judged_trials,
                "brief": str(loop_res.get("brief") or ""),
                "effective_themes": list(loop_res.get("effective_themes") or []),
                "failure_traps": list(loop_res.get("failure_traps") or []),
                "next_suggestions": list(loop_res.get("next_suggestions") or []),
            })
        except Exception as exc:  # noqa: BLE001
            _logger.warning("AI 持续学习闭环落库失败（不影响主流程返回）: %s", exc)

        loop_out["kb_records"] = n
        return loop_out

    # ------------------------------------------------------------------
    # 门槛判定 + 自动入有效策略库（端到端策略挖掘闭环，新增）
    # ------------------------------------------------------------------
    async def _gate_judge_and_promote(self, report: dict, req) -> dict:
        """门槛判定（``judge_strategy`` 规则）+ 达标自动注册有效策略库（lifecycle）。

        仅在请求携带 ``gate`` 配置时被调用；判定/入库全程失败闭合，任何异常
        只记录到返回 dict 的 ``error`` 字段，绝不影响主报告与既有功能。

        - 用复合 alpha 的 portfolio 指标（Sharpe / 最大回撤 / 前向 IC）构造策略记录；
        - ``judge_strategy(..., fallback_rules=True)`` 纯规则判定
          （verified / active / rejected）；
        - ``status == verified`` 且 ``req.promote`` → 自动注册到 lifecycle
          （IDEA → BACKTEST），6_生命周期 页面可见，可继续晋升 PAPER/LIVE。

        Returns:
            ``{"enabled", "status", "reason", "tags", "metrics", "gate",
              "promoted", "strategy_id", "promote_reasons"?, "error"?}``
        """
        from ...research.knowledge_loop import judge_strategy
        from ...paper.promotion import LifecycleManager, LifecycleState, PromotionGate
        from ...knowledge import KnowledgeStore

        gate = dict(req.gate or {})
        composite = (report.get("pipeline") or {}).get("composite") or {}
        portfolio = composite.get("portfolio") or {}
        ic_report = composite.get("ic_report") or {}

        sharpe = _num(portfolio.get("sharpe"))
        mdd = _num(portfolio.get("max_drawdown"))
        total_ret = _num(portfolio.get("total_return"))
        fwd_ic = _num(ic_report.get("ic_mean"))

        strat = {
            "run_id": req.idea,
            "state": "BACKTEST",
            "status": "",
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "composite_fwd_ic": fwd_ic,
        }
        judge = await judge_strategy(self.provider, strat, gate=gate, fallback_rules=True)

        out: Dict[str, object] = {
            "enabled": True,
            "status": judge.get("status"),
            "reason": judge.get("reason", ""),
            "tags": judge.get("tags") or [],
            "metrics": {
                "sharpe": sharpe,
                "max_drawdown": mdd,
                "total_return": total_ret,
                "fwd_ic": fwd_ic,
            },
            "gate": gate,
            "promoted": False,
            "strategy_id": "",
        }

        # 达标 → 自动注册到有效策略库（lifecycle 表）
        if judge.get("status") == "verified" and getattr(req, "promote", False):
            try:
                kb = KnowledgeStore()
                strategy_id = _e2e_strategy_id(req.idea, req.symbols or [])
                code = (report.get("strategy") or {}).get("code") or ""
                code_safe = bool((report.get("strategy") or {}).get("code_safe"))
                symbols = [s for s in (req.symbols or []) if s]
                reason = judge.get("reason", "")

                # 1) 建行（整行语义）：策略本体 + 标的信息 + 判定原因
                kb.upsert_strategy_lifecycle(
                    strategy_id,
                    idea=req.idea,
                    state="BACKTEST",
                    source="e2e",
                    code=code,
                    code_safe=code_safe,
                    symbols=symbols,
                    status="verified",
                    reason=reason,
                    brief=reason,
                )
                # 2) 补真实回测指标（部分更新，避免整行覆盖清空 sharpe）
                kb.update_strategy_state(
                    strategy_id,
                    state="BACKTEST",
                    sharpe=sharpe,
                    max_drawdown=mdd,
                    composite_fwd_ic=fwd_ic,
                    status="verified",
                    reason=reason,
                    brief=reason,
                )
                # 3) 记录晋升轨迹（IDEA → BACKTEST）
                try:
                    kb.push_strategy_transition(
                        strategy_id, "IDEA", "BACKTEST",
                        f"端到端挖掘门槛通过：{reason[:80]}")
                except Exception as exc:  # noqa: BLE001 轨迹缺失不影响入库结果
                    _logger.debug("lifecycle transition 记录失败（忽略）: %s", exc)
                out["promoted"] = True
                out["strategy_id"] = strategy_id
            except Exception as exc:  # noqa: BLE001 失败闭合：不影响主报告
                _logger.warning("端到端策略自动入库失败（不影响主报告）: %s", exc)
                out["promote_error"] = str(exc)[:200]

        return out

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
