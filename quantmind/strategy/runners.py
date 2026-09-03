"""策略运行器：用同一份策略代码，按 ``mode`` 切换运行路线。

  - backtest：批量历史回测（BacktestEngine）
  - paper：模拟交易回放（PaperEngine，实时广播事件）
  - live：实盘路由（LiveEngine -> CTP/XTP/IB 网关桩）

这就是「切换路线即可跑实盘」的落地：策略代码不变，只换 context。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..core.engine import EventEngine
from ..core.object import BarData


def _pair_round_trips(trades, cap: int = 500) -> List[dict]:
    """把成交流配对成开平回合（用于逐笔明细展示）。

    配对逻辑与 PerformanceAnalyzer._pair_pnl 一致（相邻反向成交，
    同向加仓入栈，残量回填），但保留完整的开平仓信息。最多返回
    最近 ``cap`` 个回合，避免多品种长回测时结果体积爆炸。
    """
    import dataclasses

    rounds: List[dict] = []
    stacks: Dict[str, list] = {}
    for t in trades:
        stack = stacks.setdefault(t.vt_symbol, [])
        if not stack:
            stack.append(t)
            continue
        prev = stack[-1]
        if prev.direction == t.direction:
            stack.append(t)
            continue
        stack.pop()
        direction = 1 if prev.direction.value == "多" else -1
        matched = min(prev.volume, t.volume)
        rounds.append({
            "direction": "多" if direction > 0 else "空",
            "entry_time": prev.datetime.isoformat(),
            "entry_price": round(float(prev.price), 2),
            "exit_time": t.datetime.isoformat(),
            "exit_price": round(float(t.price), 2),
            "volume": float(matched),
            "pnl": round((t.price - prev.price) * matched * direction, 2),
        })
        # 部分平仓/反手时保留残量，与 _pair_pnl 同一套语义
        if t.volume > prev.volume:
            residual = dataclasses.replace(t, volume=t.volume - prev.volume)
            stack.append(residual)
        elif prev.volume > t.volume:
            residual_open = dataclasses.replace(prev, volume=prev.volume - t.volume)
            if stack:
                stack[-1] = residual_open
            else:
                stack.append(residual_open)
    return rounds[-cap:]


def _benchmark_curve(bars: List[BarData], max_points: int = 2000) -> List[dict]:
    """买入持有基准：日频归一化收盘价（首日 = 1.0）。"""
    import pandas as pd

    if not bars:
        return []
    df = pd.DataFrame({
        "t": [b.datetime for b in bars],
        "c": [b.close_price for b in bars],
    })
    df["t"] = pd.to_datetime(df["t"], utc=True)
    daily = df.groupby(df["t"].dt.date)["c"].last()
    base = float(daily.iloc[0])
    if base <= 0:
        return []
    curve = [{"date": d.isoformat(), "nav": round(float(v) / base, 4)}
             for d, v in daily.items()]
    # 点数超限时均匀抽样（保留首尾）
    if len(curve) > max_points:
        step = len(curve) / max_points
        idx = [int(i * step) for i in range(max_points - 1)] + [len(curve) - 1]
        curve = [curve[i] for i in idx]
    return curve


def run_strategy(
    mode: str,
    strategy_class,
    vt_symbol: str,
    setting: Optional[dict],
    bars: List[BarData],
    event_engine: Optional[EventEngine] = None,
    sizes: Optional[Dict[str, float]] = None,
    gateway_name: str = "ctp",
    gateway_settings: Optional[dict] = None,
    cost=None,
    commission: Optional[float] = None,
    slippage: Optional[float] = None,
    warmup_bars: int = 0,
    daily_context=None,
    mtf_context=None,
) -> dict:
    """按模式运行策略，返回结果字典。

    重引擎（BacktestEngine/PaperEngine/LiveEngine）延迟导入以打破循环依赖。
    ``commission``/``slippage`` 仅在 ``cost`` 为空（旧式单一费率）时生效；
    传入 0 且 ``cost=False`` 即可得到干净的零成本回测，用于量化成本拖累。
    """
    # 延迟导入，避免 backtest.engine <-> strategy 循环
    from ..backtest.engine import BacktestEngine
    from ..paper.engine import PaperEngine
    from ..live.runner import LiveEngine
    from ..live import build_gateway

    data = {vt_symbol: bars}
    if mode == "backtest":
        exclude_limit = bool(setting.get("exclude_limit", False)) if setting else False
        limit_pct = (setting or {}).get("limit_pct", None)
        eng = BacktestEngine(data, sizes=sizes, event_engine=event_engine,
                             exclude_limit=exclude_limit, limit_pct=limit_pct,
                             cost_table=cost,
                             commission=commission if commission is not None else 0.0002,
                             slippage=slippage if slippage is not None else 0.0,
                             warmup_bars=warmup_bars)
        eng.add_strategy(strategy_class, vt_symbol, setting)
        # 多周期/日线级上下文注入（分钟策略查询更高周期数据，借鉴 LLM 策略挖掘实践）
        if daily_context is not None:
            eng.strategy.daily = daily_context
        if mtf_context is not None:
            eng.strategy.mtf = mtf_context
        report = eng.run()
        return {
            "mode": "backtest",
            "report": report.to_dict(),
            "equity_curve": report.equity_curve,
            "trades": report.trade_count,
            "trade_list": _pair_round_trips(eng.trades),
            "benchmark_curve": _benchmark_curve(bars),
        }

    if mode == "paper":
        eng = PaperEngine(event_engine=event_engine, sizes=sizes)
        eng.add_strategy(strategy_class, vt_symbol, setting)
        summary = eng.run_replay(data)
        return {"mode": "paper", "summary": summary, "trades": len(eng.trades)}

    if mode == "live":
        gw = build_gateway(gateway_name, event_engine or EventEngine(), gateway_settings or {})
        eng = LiveEngine(gw, event_engine=event_engine, history=data)
        strategy = strategy_class(eng, setting)
        strategy.vt_symbols = [vt_symbol]
        strategy.on_init()
        strategy.on_start()
        for bar in bars:
            strategy.on_bar(bar)
        strategy.on_stop()
        gw.close()
        return {"mode": "live", "gateway": gateway_name, "routed": True}

    raise ValueError(f"未知模式: {mode}")


def run_strategy_multi(
    strategy_class,
    vt_symbols: List[str],
    data: Dict[str, List[BarData]],
    setting: Optional[dict] = None,
    event_engine: Optional[EventEngine] = None,
    sizes: Optional[Dict[str, float]] = None,
    cost=None,
) -> dict:
    """按「多标的组合」模式运行策略（目前仅回测），返回结果字典。

    与 ``run_strategy`` 的区别：允许一次给定多个标的（``vt_symbols`` + ``data``），
    供 5 组件框架的 Universe(M5)/Portfolio(M4) 做真正的多标的组合回测。
    单标的调用退化为与 ``run_strategy("backtest", ...)`` 一致。
    """
    # 延迟导入避开循环依赖
    from ..backtest.engine import BacktestEngine

    vt_symbols = list(vt_symbols or (data.keys() if data else []))
    if not vt_symbols:
        raise ValueError("未指定标的（vt_symbols / data 为空）")
    if not all(vt in data for vt in vt_symbols):
        missing = [vt for vt in vt_symbols if vt not in data]
        raise ValueError(f"data 中缺少标的: {missing}")

    exclude_limit = bool(setting.get("exclude_limit", False)) if setting else False
    limit_pct = (setting or {}).get("limit_pct", None)
    eng = BacktestEngine(data, sizes=sizes, event_engine=event_engine,
                         exclude_limit=exclude_limit, limit_pct=limit_pct,
                         cost_table=cost)
    eng.add_strategy(strategy_class, vt_symbols[0], setting)
    eng.set_universe(vt_symbols)
    report = eng.run()
    # 实际选中的标的池（M5 过滤后）；非组合策略退化为候选集
    universe = getattr(eng.strategy, "universe_symbols", None) or list(eng.strategy.vt_symbols)
    return {
        "mode": "backtest",
        "universe": list(universe),
        "report": report.to_dict(),
        "equity_curve": report.equity_curve,
        "trades": report.trade_count,
    }
