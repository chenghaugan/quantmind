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
) -> dict:
    """按模式运行策略，返回结果字典。

    重引擎（BacktestEngine/PaperEngine/LiveEngine）延迟导入以打破循环依赖。
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
                             exclude_limit=exclude_limit, limit_pct=limit_pct)
        eng.add_strategy(strategy_class, vt_symbol, setting)
        report = eng.run()
        return {
            "mode": "backtest",
            "report": report.to_dict(),
            "equity_curve": report.equity_curve,
            "trades": report.trade_count,
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
