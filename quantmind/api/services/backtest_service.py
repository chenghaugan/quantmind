"""BacktestService: 回测 / WalkForward / 策略清单"""
import asyncio
import inspect
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple

import pandas as pd

from ...core.constant import Exchange, Interval
from ...core.contracts import default_size
from ...core.engine import EventEngine
from ...core.event import Event, EventType
from ...data.feed.base import HistoryRequest
from ...data import DataManager
from ...strategy import (
    run_strategy,
    MultiFactorStrategy,
    DualMaStrategy,
    VolTargetStrategy,
    PairTradingStrategy,
    CtaTemplate,
)
from ...strategy.components import ComposableStrategy
from ...backtest.walkforward import walk_forward
from ...research.risk_xray import compute_risk_xray
from ..schemas import BacktestRequest, WalkForwardRequest, StrategyInfo
from ..ws import manager as ws_manager


_logger = logging.getLogger("quantmind.api")


_STRATEGY_MAP = {
    "dual_ma": DualMaStrategy,
    "multifactor": MultiFactorStrategy,
    "vol_target": VolTargetStrategy,
    "pair": PairTradingStrategy,
    "composable": ComposableStrategy,
}


def _sanitize(o: Any) -> Any:
    """递归把非有限 float 转为 None，避免 JSON 序列化抛错"""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(x) for x in o]
    return o


def _strip_code_fences(source: str) -> str:
    """去除源码外围的 Markdown 代码围栏（```python ... ```），使落库/持久化的代码可编译。"""
    s = (source or "").strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines)
    return s.strip()


def _strategy_class_name(source: str) -> str:
    """从策略源码提取主类名（优先取以 Strategy 结尾的类）。"""
    import re as _re

    names = _re.findall(r"class\s+(\w+)", source or "")
    if not names:
        return ""
    for n in names:
        if n.endswith("Strategy"):
            return n
    return names[-1]



class BacktestService:
    def __init__(self, dm: DataManager, ee: EventEngine):
        self.dm = dm
        self.ee = ee
        # 动态注册的 AI 生成策略（仅实例级，避免全局污染）
        self._extra_strategies: Dict[str, type] = {}
        # 耐重启：启动时把 knowledge.db 已沉淀/已注册策略载入运行池
        self._load_persisted_strategies()

    def register_generated_strategy(self, name: str, source: str) -> Tuple[bool, str, dict]:
        """注册 AI 生成策略源码 -> 实例级策略池。

        先经沙箱二次校验（compile_strategy），通过后才在隔离命名空间 exec，
        从其中找 CtaTemplate 子类存入 ``self._extra_strategies``。

        :return: (ok, err, info)；info = {"name", "parameters"}
        """
        from ...ai.sandbox import compile_strategy

        source = _strip_code_fences(source)

        ok, err, _ = compile_strategy(source)
        if not ok:
            return False, err or "代码未通过沙箱校验", {}
        try:
            ns: Dict = {}
            exec(compile(source, "<generated>", "exec"), ns, ns)
        except Exception as exc:  # noqa: BLE001
            return False, f"执行策略源码失败: {exc}", {}
        cls = None
        for v in ns.values():
            if inspect.isclass(v) and issubclass(v, CtaTemplate) and v is not CtaTemplate:
                cls = v
                break
        if cls is None:
            # 兜底：放宽到名字以 Strategy 结尾的类
            for v in ns.values():
                if inspect.isclass(v) and (v.__name__.endswith("Strategy") or "Strategy" in v.__name__):
                    cls = v
                    break
        if cls is None:
            return False, "未找到策略类", {}
        self._extra_strategies[name] = cls
        return True, "", {"name": name, "parameters": list(getattr(cls, "parameters", []))}

    def _load_persisted_strategies(self) -> None:
        """启动/重建时把已沉淀策略载入运行池（耐重启）。

        来源分两类：
        1. 规范适配模块 ``quantmind.strategy.mined`` 的 ``MINED_STRATEGIES``（历史挖掘策略的可运行重写）；
        2. knowledge.db 中已注册/挖掘的策略（``lifecycle`` 表 + ``strategies`` 表）。
        历史代码可能因模块路径漂移编译失败 -> 跳过，不影响其它策略加载。
        """
        # 1) 规范适配策略：稳定可运行，始终注册（按类名）
        try:
            from ...strategy.mined import MINED_STRATEGIES as _mined

            seen = set(self._extra_strategies)
            for name, cls in _mined.items():
                if name in seen:
                    continue
                self._extra_strategies[name] = cls
                seen.add(name)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("加载规范适配策略失败(忽略): %s", exc)

        if not self._extra_strategies:
            return
        seen = set(self._extra_strategies)
        try:
            from ...knowledge.store import KnowledgeStore

            ks = KnowledgeStore()
            # 2) 已注册生命周期策略（有明确名称）
            for lc in ks.list_strategy_lifecycles(limit=300):
                name = (lc.get("strategy_id") or "").strip()
                code = lc.get("code") or ""
                if not name or not code or name in seen:
                    continue
                ok, _, _ = self.register_generated_strategy(name, code)
                if ok:
                    seen.add(name)
            # 3) 研究挖掘脚本（从类名取名）
            for rec in ks.list_mined_strategies(limit=300):
                name = _strategy_class_name(rec.get("code") or "")
                code = rec.get("code") or ""
                if not name or name in seen or not code:
                    continue
                ok, _, _ = self.register_generated_strategy(name, code)
                if ok:
                    seen.add(name)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("加载已沉淀策略失败(忽略): %s", exc)

    def _resolve_strategy_class(self, name: str):
        """按名解析策略类：内置映射 -> 运行池 -> 知识库惰性加载 -> 规范适配模块。"""
        cls = _STRATEGY_MAP.get(name) or self._extra_strategies.get(name)
        if cls is not None:
            return cls
        # 惰性：从库中按名载入（例如 list 已展示但尚未进池的挖掘/注册策略）
        try:
            from ...knowledge.store import KnowledgeStore

            ks = KnowledgeStore()
            for lc in ks.list_strategy_lifecycles(limit=300):
                if (lc.get("strategy_id") or "").strip() == name:
                    ok, _, _ = self.register_generated_strategy(name, lc.get("code") or "")
                    return self._extra_strategies.get(name)
            for rec in ks.list_mined_strategies(limit=300):
                if _strategy_class_name(rec.get("code") or "") == name:
                    ok, _, _ = self.register_generated_strategy(name, rec.get("code") or "")
                    return self._extra_strategies.get(name)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("按名加载策略失败(忽略): %s", exc)
        # 兜底：规范适配模块
        try:
            from ...strategy.mined import MINED_STRATEGIES as _mined

            cls = _mined.get(name)
            if cls is not None:
                self._extra_strategies[name] = cls
                return cls
        except Exception:  # noqa: BLE001
            pass
        return None

    def list_strategies(self) -> List[StrategyInfo]:
        result = [
            StrategyInfo(
                name="dual_ma",
                description="双均线趋势/动量策略",
                parameters={"fast": 5, "slow": 20, "size": 1, "max_pos": 1.0},
            ),
            StrategyInfo(
                name="multifactor",
                description="多因子组合策略（动量+均值回复+波动率）",
                parameters={"specs": "see research", "threshold": 0.3, "size": 1, "max_pos": 1.0},
            ),
            StrategyInfo(
                name="vol_target",
                description="全天候风格：波动率目标+动量过滤(单标的风险平价)",
                parameters={"lookback": 20, "target_vol": 0.20, "momentum_win": 60, "size": 1, "max_pos": 1.0},
            ),
            StrategyInfo(
                name="pair",
                description="配对交易：价差合成标的 z-score 均值回复",
                parameters={"window": 30, "entry_z": 1.5, "exit_z": 0.3, "size": 1, "max_pos": 1.0},
            ),
            StrategyInfo(
                name="composable",
                description="5 组件可组合策略（Alpha/Portfolio/Risk/Execution 可插拔）",
                parameters={"alpha": "MultiFactorAlpha/MomentumAlpha", "risk": "NullRisk/RiskGateModel"},
            ),
        ]
        for k in self._extra_strategies:
            result.append(StrategyInfo(name=k, description="AI 生成策略", parameters={"size": 1}))
        # 追加研究挖掘的策略（即使历史代码当前不可编译，也展示供查看/重建）
        seen = {s.name for s in result}
        try:
            from ...knowledge.store import KnowledgeStore

            ks = KnowledgeStore()
            for rec in ks.list_mined_strategies(limit=100):
                name = _strategy_class_name(rec.get("code") or "")
                if not name or name in seen:
                    continue
                seen.add(name)
                result.append(StrategyInfo(
                    name=name,
                    description=("AI挖掘:" + str(rec.get("idea") or "")[:40])
                    if rec.get("idea") else "AI 挖掘策略",
                    parameters={"size": 1, "source": "mined"},
                ))
        except Exception:  # noqa: BLE001
            pass
        return result

    async def run_backtest(self, req: BacktestRequest) -> Dict[str, Any]:
        strat_class = self._resolve_strategy_class(req.strategy) or MultiFactorStrategy
        vt = f"{req.symbol}.{req.exchange.upper()}"

        # 发送开始事件
        await ws_manager.broadcast({
            "type": "backtest_start",
            "strategy": req.strategy,
            "symbol": req.symbol,
            "exchange": req.exchange,
            "mode": req.mode,
            "timestamp": datetime.now().isoformat(),
        })
        
        try:
            bars = await self.dm.get_bar_data(
                HistoryRequest(
                    symbol=req.symbol,
                    exchange=Exchange(req.exchange.upper()),
                    interval=Interval("1d"),
                )
            )
            if not bars:
                _logger.warning(f"回测无数据: {req.symbol}.{req.exchange}")
                await ws_manager.broadcast({
                    "type": "backtest_error",
                    "strategy": req.strategy,
                    "error": "无数据",
                })
                return {"error": "无数据"}
            
            # 发送进度事件（数据加载完成）
            await ws_manager.broadcast({
                "type": "backtest_progress",
                "strategy": req.strategy,
                "progress": 0.3,
                "message": f"已加载 {len(bars)} 根K线",
            })
            
            sizes = dict(req.sizes) or {vt: default_size(vt)}
            result = await asyncio.to_thread(
                run_strategy,
                req.mode,
                strat_class,
                vt,
                dict(req.setting),
                bars,
                self.ee,
                sizes,
                req.gateway,
                None,
                req.cost,
            )
            _logger.info(f"回测完成: {req.strategy} on {vt}, {len(bars)} bars")
            
            # 计算风险 X 光指标
            result_sanitized = _sanitize(result)
            try:
                equity_curve = _extract_equity_curve(result)
                # trades/positions 可能为计数而非明细，归一化为 compute_risk_xray 所需结构
                _raw_trades = result.get("trades", [])
                trades = _raw_trades if isinstance(_raw_trades, list) else []
                _raw_pos = result.get("positions", {})
                positions = _raw_pos if isinstance(_raw_pos, dict) else {}
                if equity_curve is not None and len(equity_curve) > 0:
                    risk_xray = compute_risk_xray(equity_curve, trades, positions)
                    result_sanitized["risk_xray"] = risk_xray.to_dict()
                    _logger.info(f"风险 X 光已生成: 夏普={risk_xray.sharpe_ratio:.2f}, 最大回撤={risk_xray.max_drawdown:.2%}")
            except Exception as exc:  # noqa: BLE001
                _logger.warning(f"风险 X 光计算失败: {exc}")
            
            # 发送完成事件
            await ws_manager.broadcast({
                "type": "backtest_complete",
                "strategy": req.strategy,
                "progress": 1.0,
                "message": "回测完成",
                "summary": result_sanitized.get("summary", {}),
            })
            
            return result_sanitized
        except Exception as e:
            _logger.error(f"回测失败: {req.strategy} on {vt} - {str(e)}", exc_info=True)
            await ws_manager.broadcast({
                "type": "backtest_error",
                "strategy": req.strategy,
                "error": str(e),
            })
            return {"error": f"回测失败: {str(e)}"}

    async def run_walkforward(self, req: WalkForwardRequest) -> Dict[str, Any]:
        """Walk-Forward 滚动样本外验证"""
        strat_class = self._resolve_strategy_class(req.strategy) or MultiFactorStrategy
        vt = f"{req.symbol}.{req.exchange.upper()}"

        # 自动计算需要的历史数据长度
        step = req.step or req.test_window
        min_bars_needed = req.train_window + req.test_window * 2
        days_needed = max(min_bars_needed + 50, 500)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_needed)

        bars = await self.dm.get_bar_data(
            HistoryRequest(
                symbol=req.symbol,
                exchange=Exchange(req.exchange.upper()),
                interval=Interval("1d"),
                start=start_date,
                end=end_date,
            )
        )
        if not bars:
            return {"error": "无数据"}

        min_required = req.train_window + req.test_window
        if len(bars) < min_required:
            return {
                "error": f"样本不足：需要至少 {min_required} 根，当前仅 {len(bars)} 根。"
                f"请减小 train_window/test_window。"
            }

        sizes = {vt: default_size(vt)}
        try:
            result = await asyncio.to_thread(
                walk_forward,
                bars,
                strat_class,
                dict(req.setting),
                vt,
                req.train_window,
                req.test_window,
                req.step,
                sizes,
                req.capital,
                req.cost if req.cost else None,
            )
            return _sanitize(result.to_dict())
        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"运行失败: {str(e)}"}

    async def run_paper(self, req: "PaperRunRequest") -> Dict[str, Any]:
        """模拟盘实跑：把策略（含 AI 注册策略）部署到 PaperEngine 做历史回放。

        复用 ``run_strategy(mode="paper")`` 的同一套策略代码与撮合模型，
        产出与实盘同源的 PaperEngine 回放结果（现金/持仓/成交/权益），
        供生命周期晋升到 PAPER 及前端「模拟盘实跑」闭环使用。
        """
        strat_class = self._resolve_strategy_class(req.strategy)
        if strat_class is None:
            return {"error": f"策略不存在或未注册: {req.strategy}"}
        vt = f"{req.symbol}.{req.exchange.upper()}"

        # 发送开始事件
        await ws_manager.broadcast({
            "type": "backtest_start",
            "strategy": req.strategy,
            "symbol": req.symbol,
            "exchange": req.exchange,
            "mode": "paper",
            "timestamp": datetime.now().isoformat(),
        })

        end_date = datetime.now()
        start_date = end_date - timedelta(days=req.days or 400)
        try:
            bars = await self.dm.get_bar_data(
                HistoryRequest(
                    symbol=req.symbol,
                    exchange=Exchange(req.exchange.upper()),
                    interval=Interval("1d"),
                    start=start_date,
                    end=end_date,
                )
            )
        except Exception as exc:  # noqa: BLE001
            await ws_manager.broadcast({
                "type": "backtest_error",
                "strategy": req.strategy,
                "error": f"取数失败: {exc}",
            })
            return {"error": f"取数失败: {exc}"}
        if not bars:
            await ws_manager.broadcast({
                "type": "backtest_error",
                "strategy": req.strategy,
                "error": "无数据",
            })
            return {"error": "无数据"}

        # 发送进度事件
        await ws_manager.broadcast({
            "type": "backtest_progress",
            "strategy": req.strategy,
            "progress": 0.3,
            "message": f"已加载 {len(bars)} 根K线",
        })

        sizes = {vt: default_size(vt)}
        try:
            result = await asyncio.to_thread(
                run_strategy,
                "paper",
                strat_class,
                vt,
                dict(req.setting),
                bars,
                self.ee,
                sizes,
                "ctp",
                None,
                None,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.error(f"模拟盘实跑失败: {req.strategy} on {vt} - {exc}", exc_info=True)
            await ws_manager.broadcast({
                "type": "backtest_error",
                "strategy": req.strategy,
                "error": f"模拟盘实跑失败: {exc}",
            })
            return {"error": f"模拟盘实跑失败: {exc}"}

        summary = result.get("summary", {})
        # 用权益曲线算真实夏普/回撤（参考 validation.py 年化 252 公式）；缺权益则回退纯纸面
        _eq = _extract_equity_curve(result)
        _eq_metrics = _metrics_from_equity(_eq) if _eq is not None else {}
        out = {
            "ok": True,
            "strategy": req.strategy,
            "vt_symbol": vt,
            "bars": len(bars),
            "trade_count": result.get("trades", 0),
            "cash": summary.get("cash"),
            "positions": summary.get("positions", {}),
            "metrics": {
                "trade_count": result.get("trades", 0),
                "final_cash": summary.get("cash"),
                "open_positions": len(summary.get("positions", {})),
                "sharpe": _eq_metrics.get("sharpe"),
                "max_drawdown": _eq_metrics.get("max_drawdown"),
                "total_return": _eq_metrics.get("total_return"),
            },
        }
        
        # 发送完成事件
        await ws_manager.broadcast({
            "type": "backtest_complete",
            "strategy": req.strategy,
            "progress": 1.0,
            "message": "模拟盘实跑完成",
            "summary": summary,
        })
        
        return _sanitize(out)


def _extract_equity_curve(result: Dict[str, Any]) -> pd.Series:
    """从回测结果中提取权益曲线（归一化为数值 Series）。

    equity_curve 可能为 ``[{date, equity...}, ...]`` 的 dict 列表，
    也可能为纯数值列表；这里统一转成 float Series，供指标计算使用。
    """
    raw = None
    if "equity_curve" in result:
        raw = result["equity_curve"]
    elif "portfolio" in result and "equity" in result["portfolio"]:
        raw = result["portfolio"]["equity"]
    elif "summary" in result and "equity_curve" in result["summary"]:
        raw = result["summary"]["equity_curve"]
    if raw is None:
        return pd.Series(dtype=float)

    series = pd.Series(raw)
    # 若元素为 dict，抽取数值字段；否则按原值处理
    if series.notna().any() and isinstance(series.dropna().iloc[0], dict):
        first = series.dropna().iloc[0]
        key = next((k for k in ("equity", "value", "balance", "nav") if k in first), None)
        if key is not None:
            series = series.apply(lambda d: d.get(key) if isinstance(d, dict) else d)
    return pd.to_numeric(series, errors="coerce")

def _metrics_from_equity(equity_curve):
    """从权益曲线算出年化夏普(252)/最大回撤/总收益（纯 pandas，无新依赖）。

    参考 backtest/validation.py 年化公式：mean/std * sqrt(252)。
    权益曲线过短或全为 NaN 时返回空 dict（调用方回退纸面判读）。
    """
    if equity_curve is None or getattr(equity_curve, "empty", True):
        return {}
    s = pd.Series(equity_curve).dropna()
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 2 or float(s.iloc[0]) == 0:
        return {}
    rets = s.pct_change().dropna()
    std = rets.std()
    sharpe = float((rets.mean() / std * (252 ** 0.5))) if std and std > 0 else 0.0
    running_max = s.cummax()
    dd = s / running_max - 1.0
    mdd = float(dd.min())
    total = float((s.iloc[-1] / s.iloc[0]) - 1.0)
    return {"sharpe": sharpe, "max_drawdown": mdd, "total_return": total}
