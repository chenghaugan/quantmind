"""CLI 入口（Typer + Rich）。

自动化/定时任务通道；与 Web 共用同一套引擎与数据层。
命令：
  quantmind smoke   端到端冒烟：拉取多资产数据并入库读出
  quantmind factor   计算并评估一个因子（IC/IR/衰减）
  quantmind backtest 回测 / 模拟 / 实盘（按 --mode）
  quantmind research AI 研究：idea -> 规格/因子/策略代码
  quantmind e2e      完整端到端演示（数据→因子→多因子策略→回测→模拟→实盘→AI）
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

import typer
from rich.console import Console
from rich.table import Table

from .config import get_settings
from .data import build_default_registry, DataManager, InMemoryStore
from .data.feed.base import HistoryRequest
from .core.constant import Exchange, Interval
from .core.engine import EventEngine
from .core.contracts import default_size
from .research import MomentumFactor, FactorEvaluator, FactorSpec, build_model_from_specs, build_factor_registry, eval_factor_expression
from .strategy import run_strategy, MultiFactorStrategy, DualMaStrategy, VolTargetStrategy, PairTradingStrategy, build_spread_bars
from .ai import ResearchAgent
from .monitoring import Notifier

app = typer.Typer(help="QuantMind 量化投研框架 CLI")
console = Console()
_logger = logging.getLogger("quantmind.cli")

SMOKE_CASES = [
    ("rb0", Exchange.SHFE, Interval.DAILY, "商品期货主连(螺纹)"),
    ("IF0", Exchange.CFFEX, Interval.DAILY, "金融期货主连(沪深300)"),
    ("600519", Exchange.SSE, Interval.DAILY, "A股(贵州茅台)"),
    ("00700", Exchange.HKEX, Interval.DAILY, "港股(腾讯)"),
    ("IO2409-C-3900", Exchange.CFFEX, Interval.DAILY, "期权(沪深300股指期权)"),
]


def _make_dm():
    s = get_settings()
    root = s.local_data_root or None
    registry = build_default_registry(local_data_root=root)
    store = InMemoryStore()
    return DataManager(registry, store)


@app.command()
def info() -> None:
    """打印配置与已注册数据源。"""
    s = get_settings()
    table = Table(title="QuantMind 配置")
    table.add_column("项"); table.add_column("值")
    for k, v in [("db_url", s.db_url), ("redis_url", s.redis_url),
                 ("llm_provider", s.llm_provider), ("api_url", s.api_url),
                 ("local_data_root", s.local_data_root or "(未配置)")]:
        table.add_row(k, v)
    srcs = ("china_futures_csv(本地,优先) / akshare_future / mootdx_astock / em_hk / "
            "akshare_option / mock(兜底)") if s.local_data_root else \
           "akshare_future / mootdx_astock / em_hk / akshare_option / mock(兜底)"
    console.print("[green]数据源：[/green]", srcs)


@app.command()
def smoke() -> None:
    """端到端冒烟：拉取多资产数据并入库读出。"""
    asyncio.run(_smoke())


@app.command()
def factor(
    symbol: str = typer.Option("rb0", help="合约代码"),
    exchange: str = typer.Option("SHFE"),
    name: str = typer.Option("momentum_20", help="因子名或表达式(用 --expr)"),
    expr: str = typer.Option(None, help="因子表达式，如 (close/ref(close,60)-1)"),
    window: int = typer.Option(20),
    years: int = typer.Option(1),
) -> None:
    """计算并评估因子（IC/IR/衰减/分位收益）。"""
    asyncio.run(_factor(symbol, exchange, name, expr, window, years))


@app.command()
def backtest(
    symbol: str = typer.Option("rb0"),
    exchange: str = typer.Option("SHFE"),
    strategy: str = typer.Option("multifactor", help="dual_ma | multifactor | vol_target | pair"),
    mode: str = typer.Option("backtest", help="backtest | paper | live"),
    gateway: str = typer.Option("ctp"),
    years: int = typer.Option(1),
    exclude_limit: bool = typer.Option(False, help="剔除涨跌停日成交（A股/港股严谨回测）"),
    limit_pct: float = typer.Option(0.10, help="涨跌停幅度阈值（与 exclude_limit 配合使用）"),
    leg2: str = typer.Option(None, help="配对交易第二腿合约，如 hc0.SHFE（仅 strategy=pair 用）"),
) -> None:
    """回测 / 模拟 / 实盘（同一策略按 --mode 切换路线）。"""
    asyncio.run(_backtest(symbol, exchange, strategy, mode, gateway, years, exclude_limit, limit_pct, leg2))


@app.command()
def research(idea: str = typer.Argument(..., help="投资想法（自然语言）"),
             asset_class: str = typer.Option("", help="资产类别")) -> None:
    """AI 研究：idea -> 研究规格 / 候选因子 / 策略代码。"""
    asyncio.run(_research(idea, asset_class))


@app.command()
def e2e() -> None:
    """完整端到端演示。"""
    asyncio.run(_e2e())


# ---- 实现 ----
async def _smoke() -> None:
    dm = _make_dm()
    await dm.connect()
    end = datetime.now(); start = end - timedelta(days=30)
    table = Table(title="冒烟结果（近 30 日）")
    table.add_column("描述"); table.add_column("vt_symbol"); table.add_column("根数"); table.add_column("末价")
    for symbol, exch, interval, desc in SMOKE_CASES:
        try:
            bars = await dm.get_bar_data(HistoryRequest(symbol=symbol, exchange=exch, interval=interval, start=start, end=end))
            last = bars[-1].close_price if bars else float("nan")
            table.add_row(desc, f"{symbol}.{exch.value}", str(len(bars)), f"{last:.2f}")
        except Exception as exc:  # noqa: BLE001
            table.add_row(desc, f"{symbol}.{exch.value}", "FAIL", str(exc)[:40])
    console.print(table)
    await dm.close()


async def _fetch(symbol, exchange, years):
    dm = _make_dm(); await dm.connect()
    end = datetime.now(); start = end - timedelta(days=365 * years)
    bars = await dm.get_bar_data(HistoryRequest(symbol=symbol, exchange=Exchange(exchange.upper()),
                                                interval=Interval.DAILY, start=start, end=end))
    await dm.close()
    return bars


async def _factor(symbol, exchange, name, expr, window, years) -> None:
    bars = await _fetch(symbol, exchange, years)
    if not bars:
        console.print("[red]无数据[/red]"); return
    if expr:
        from .research.factors.base import bars_to_df
        series = eval_factor_expression(expr, bars_to_df(bars)); fname = expr
    else:
        f = build_factor_registry().get(name) if name in [x["name"] for x in build_factor_registry().list_factors()] \
            else __import__("quantmind.research.technical", fromlist=["build_factor"]).build_factor(name.split("_")[0], window)
        series = f.compute(bars); fname = f.meta.name
    series.name = fname
    rep = FactorEvaluator().evaluate(series, bars)
    console.print(f"[bold]因子 {fname}[/bold] 样本 {rep.n_samples}")
    console.print(f"IC(rank)={rep.ic_mean:.4f}  IC(pearson)={rep.ic_pearson:.4f}  IR={rep.ir:.4f}  正向比={rep.ic_positive_ratio:.3f}")
    console.print(f"IC衰减(1~5日): {[round(x,3) if x==x else None for x in rep.ic_decay]}  半衰期={rep.ic_decay_half_life}")
    console.print(f"IC 95% CI: [{rep.ic_ci_low}, {rep.ic_ci_high}]")
    console.print(f"多头组收益={rep.top_quantile_return:.5f}  多空收益={rep.long_short_return:.5f}")
    console.print(f"单调性(5/10组)={rep.monotonicity_5:.3f}/{rep.monotonicity_10:.3f}  年化换手={rep.turnover_annual:.2f}")
    console.print(f"多空组合: 收益={rep.ls_portfolio_return:.4f}  Sharpe={rep.ls_portfolio_sharpe:.3f}  MDD={rep.ls_portfolio_mdd:.4f}")
    console.print(f"[bold green]综合主分={rep.composite_score:.3f}[/bold green]")


async def _backtest(symbol, exchange, strategy, mode, gateway, years, exclude_limit=False, limit_pct=0.10, leg2=None) -> None:
    bars = await _fetch(symbol, exchange, years)
    if not bars:
        console.print("[red]无数据[/red]"); return
    vt = f"{symbol}.{exchange.upper()}"
    sizes = {vt: default_size(vt)}
    strat_class = {"multifactor": MultiFactorStrategy, "dual_ma": DualMaStrategy,
                   "vol_target": VolTargetStrategy, "pair": PairTradingStrategy}.get(strategy, MultiFactorStrategy)
    setting = {"size": sizes[vt], "max_pos": 1.0}
    if exclude_limit:
        setting["exclude_limit"] = True
        setting["limit_pct"] = limit_pct
    # 配对交易：构造价差合成标的（leg1 - leg2）
    if strategy == "pair":
        if not leg2:
            leg2 = "hc0.SHFE"
        sym2, ex2 = leg2.split(".")
        bars2 = await _fetch(sym2, ex2, years)
        if not bars2:
            console.print("[red]配对第二腿无数据[/red]"); return
        bars = build_spread_bars(bars, bars2)
        vt = f"SPREAD.{bars[0].exchange.value}"
        sizes = {vt: 1}
        setting = {"size": 1, "max_pos": 1.0}
    ee = EventEngine(); await ee.start()
    Notifier().attach(ee)
    res = run_strategy(mode, strat_class, vt, setting, bars, ee, sizes, gateway_name=gateway)
    await ee.stop()
    console.print(f"[bold]{mode.upper()} 结果 ({strategy})[/bold]")
    console.print(res)
    if mode == "live":
        console.print(f"[green]已切换至实盘路线（{gateway} 网关桩），委托已路由[/green]")


async def _research(idea, asset_class) -> None:
    agent = ResearchAgent()
    out = await agent.research(idea, asset_class)
    console.print(f"[bold]研究：{idea}[/bold]")
    console.print(f"资产类别: {out.spec.asset_class}")
    console.print(f"假设: {out.spec.hypothesis}")
    console.print(f"候选因子: {[f.name for f in out.factors]}")
    console.print(f"风险要点: {out.spec.risk_notes}")
    console.print(f"生成策略代码安全: {out.code_safe}  错误: {out.code_errors}")


async def _e2e() -> None:
    console.print("[bold cyan]==== QuantMind 端到端演示 ====[/bold cyan]")
    dm = _make_dm(); await dm.connect()
    ee = EventEngine(); await ee.start()
    evt_count = {"n": 0}
    async def cap(e): evt_count["n"] += 1
    ee.register_general(cap)
    Notifier().attach(ee)

    symbol, exchange, years = "rb0", "SHFE", 1
    vt = f"{symbol}.{exchange}"
    end = datetime.now(); start = end - timedelta(days=365 * years)
    bars = await dm.get_bar_data(HistoryRequest(symbol=symbol, exchange=Exchange(exchange), interval=Interval.DAILY, start=start, end=end))
    console.print(f"[green]1) 历史数据[/green]: 获取 {symbol} {len(bars)} 根日线")

    f = MomentumFactor(20); series = f.compute(bars); series.name = f.meta.name
    rep = FactorEvaluator().evaluate(series, bars)
    console.print(f"[green]2) 动量因子评估[/green]: IC={rep.ic_mean:.4f} IR={rep.ir:.4f} 多头收益={rep.top_quantile_return:.5f}")

    specs = [FactorSpec(name="mom", kind="momentum", window=20, weight=1.0),
             FactorSpec(name="rev", kind="mean_reversion", window=60, weight=-0.5),
             FactorSpec(name="vol", kind="volatility", window=20, weight=-0.3)]
    model = build_model_from_specs(specs, bars)
    target = model.target_position(bars, size=default_size(vt), max_pos=1.0)
    console.print(f"[green]3) 多因子组合[/green]: 目标仓位非空 {int((target != 0).sum())}/{len(target)} 根")

    sizes = {vt: default_size(vt)}; setting = {"size": sizes[vt], "max_pos": 1.0}
    for mode in ("backtest", "paper", "live"):
        res = run_strategy(mode, MultiFactorStrategy, vt, setting, bars, ee, sizes, gateway_name="ctp")
        tag = {"backtest": "回测", "paper": "模拟", "live": "实盘(CTP桩)"}[mode]
        console.print(f"[green]4) {tag}[/green]: {res.get('report', res.get('summary', res.get('routed')))}")

    # 策略同步运行会把事件入队；让事件循环有机会分发后再统计（API 常驻进程无需此步）。
    await asyncio.sleep(0)

    out = await ResearchAgent().research("螺纹钢期货动量与期限结构因子组合", "期货")
    console.print(f"[green]5) AI 研究[/green]: 因子 {[x.name for x in out.factors]} 代码安全={out.code_safe}")

    # 再让一轮事件循环把 AI/日志事件也分发完
    await asyncio.sleep(0)
    await ee.stop()
    await dm.close()
    console.print(f"[bold green]事件总线共分发 {evt_count['n']} 个事件（驱动 Web 实时推送）[/bold green]")


if __name__ == "__main__":
    app()
