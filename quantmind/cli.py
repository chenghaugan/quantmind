"""CLI 入口（Typer + Rich）。

自动化/定时任务通道；与 Web 共用同一套引擎与数据层。
命令：
  quantmind smoke   端到端冒烟：拉取多资产数据并入库读出
  quantmind factor   计算并评估一个因子（IC/IR/衰减）
  quantmind cs       多标的截面因子评估（严格截面 rank）
  quantmind backtest 回测 / 模拟 / 实盘（按 --mode）
  quantmind research AI 研究：idea -> 规格/因子/策略代码
  quantmind e2e      完整端到端演示（数据→因子→多因子策略→回测→模拟→实盘→AI）
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

import pandas as pd
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
from .research.factors.alpha_cs import Panel, list_alpha_cs
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
    registry = build_default_registry(
        local_data_root=s.local_data_root or None,
        local_stock_root=s.local_stock_root or None,
        local_hk_root=s.local_hk_root or None,
        local_option_root=s.local_option_root or None,
    )
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
                 ("local_data_root", s.local_data_root or "(未配置)"),
                 ("local_stock_root", s.local_stock_root or "(未配置)"),
                 ("local_hk_root", s.local_hk_root or "(未配置)"),
                 ("local_option_root", s.local_option_root or "(未配置)"),
                 ("seat_data_root", s.seat_data_root or "(未配置)")]:
        table.add_row(k, v)
    console.print(table)
    has_local = any([s.local_data_root, s.local_stock_root, s.local_hk_root, s.local_option_root])
    srcs = ("china_futures_csv(本地,优先) / china_astock_parquet / china_hk_parquet / "
            "china_option_parquet / akshare_future / mootdx_astock / em_hk / "
            "akshare_option / mock(兜底)") if has_local else \
        "akshare_future / mootdx_astock / em_hk / akshare_option / mock(兜底)"
    console.print("[green]数据源：[/green]", srcs)
    if s.seat_data_root:
        console.print("[green]席位因子源：[/green] TradingAgents_for_Futures 排名 CSV（F1–F8 可用，限商品期货）")


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
def cs(
    symbols: str = typer.Option("rb0,hc0,bu0,i0", help="多标的，逗号分隔，如 rb0,hc0,bu0,i0"),
    exchange: str = typer.Option("SHFE", help="所有标的共用交易所"),
    name: str = typer.Option("alpha021", help="截面 Alpha 因子名（alpha002..alpha101 / alpha191_*）"),
    years: int = typer.Option(1),
    bt: bool = typer.Option(False, help="同时把该截面因子跑成多空组合回测（研究与回测闭环）"),
) -> None:
    """多标的截面因子评估（严格截面 rank）：构建面板→算因子→截面 IC/组合。

    与 ``factor``（单标的滚动近似）不同，此处对每个交易日横截面上对所有标的做 rank，
    得到 WorldQuant 公式本意的截面因子，再算截面 Spearman IC 与多空组合。需 ≥2 标的。
    加 --bt 会把该因子直接转成每日横截面多空组合的回测绩效。
    """
    asyncio.run(_cs(symbols, exchange, name, years, bt))


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
    cost: bool = typer.Option(False, help="启用真实成本模型（按品种差异化费率/平今/印花税/保证金）"),
    leg2: str = typer.Option(None, help="配对交易第二腿合约，如 hc0.SHFE（仅 strategy=pair 用）"),
) -> None:
    """回测 / 模拟 / 实盘（同一策略按 --mode 切换路线）。"""
    asyncio.run(_backtest(symbol, exchange, strategy, mode, gateway, years, exclude_limit, limit_pct, cost, leg2))


@app.command()
def research(idea: str = typer.Argument(..., help="投资想法（自然语言）"),
             asset_class: str = typer.Option("", help="资产类别")) -> None:
    """AI 研究：idea -> 研究规格 / 候选因子 / 策略代码。"""
    asyncio.run(_research(idea, asset_class))


@app.command()
def seat(
    symbol: str = typer.Option("RB", help="期货品种代码（如 RB/CU/AG）"),
    exchange: str = typer.Option("SHFE", help="关联价格所用交易所（用于算 IC）"),
    root: str = typer.Option("", help="席位数据根目录；默认用 QM_SEAT_DATA_ROOT 或 <local_data_root>/qihuo/database/positioning"),
) -> None:
    """期货席位因子 F1–F8（对接 TradingAgents_for_Futures 仓库排名 CSV）。

    需先 clone 该仓库并把 seat_data_root 指向 qihuo/database/positioning。
    计算净持仓矩阵 → F1–F8，并尝试用本地期货价格算各因子与次日收益的 IC。
    """
    asyncio.run(_seat(symbol, exchange, root))


@app.command()
def e2e() -> None:
    """完整端到端演示。"""
    asyncio.run(_e2e())


@app.command()
def wf(
    symbol: str = typer.Option("rb0"),
    exchange: str = typer.Option("SHFE"),
    strategy: str = typer.Option("multifactor", help="dual_ma | multifactor | vol_target | pair"),
    years: int = typer.Option(3),
    train_window: int = typer.Option(250, help="训练/预热窗口（根）"),
    test_window: int = typer.Option(60, help="每折测试窗口（根）"),
    step: int = typer.Option(60, help="滚动步长（根）；默认=test_window 不重叠"),
    cost: bool = typer.Option(False, help="启用真实成本模型"),
) -> None:
    """Walk-forward 滚动样本外验证：把历史切成多折，给出样本外稳定性与过拟合预警。"""
    asyncio.run(_wf(symbol, exchange, strategy, years, train_window, test_window, step, cost))


@app.command()
def risk(
    profile: str = typer.Option("default", help="限额档：default | conservative | unlimited(仅测试)"),
    symbol: str = typer.Option("rb2410", help="合约代码"),
    exchange: str = typer.Option("SHFE"),
    volume: float = typer.Option(5.0, help="示例委托手数"),
    price: float = typer.Option(3500.0, help="示例委托价格（0=市价）"),
    equity: float = typer.Option(1_000_000.0, help="账户权益（用于保证金/亏损熔断判断）"),
) -> None:
    """风控体检：打印限额档、当前交易时段，并对一笔示例委托做拒/放行试算。"""
    asyncio.run(_risk_report(profile, symbol, exchange, volume, price, equity))


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


async def _cs(symbols, exchange, name, years, bt=False) -> None:
    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    if len(syms) < 2:
        console.print("[red]截面评估需要至少 2 个标的[/red]"); return
    if name not in list_alpha_cs():
        console.print(f"[red]未知截面因子 {name}[/red]；可用: {list_alpha_cs()[:6]}...")
        return
    dm = _make_dm(); await dm.connect()
    end = datetime.now(); start = end - timedelta(days=365 * years)
    bars_by_symbol: Dict[str, list] = {}
    for sym in syms:
        try:
            bars = await dm.get_bar_data(HistoryRequest(symbol=sym, exchange=Exchange(exchange.upper()),
                                                       interval=Interval.DAILY, start=start, end=end))
            if bars:
                bars_by_symbol[sym] = bars
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]{sym} 取数失败: {exc}[/yellow]")
    await dm.close()
    if len(bars_by_symbol) < 2:
        console.print("[red]可用标的不足 2 个，无法做截面[/red]"); return
    panel = Panel.from_bars(bars_by_symbol)
    if panel.close.empty:
        console.print("[red]面板为空[/red]"); return
    evaluator = FactorEvaluator()
    reports = evaluator.evaluate_cross_sectional_panel([name], panel)
    rep = reports.get(name)
    if rep is None:
        console.print(f"[red]因子 {name} 未计算[/red]"); return
    console.print(f"[green]多标的截面因子评估[/green] {name} @ {exchange}  "
                  f"标的数={len(panel.symbols)} 截面数={rep.n_samples}")
    table = Table(title=f"截面因子 {name}")
    table.add_column("指标"); table.add_column("值")
    for k, v in rep.to_dict().items():
        table.add_row(k, str(v))
    console.print(table)

    if bt:
        from .research.cross_sectional_backtest import cross_sectional_backtest
        res = cross_sectional_backtest(panel, name, forward_periods=1, n_groups=5)
        p = res["portfolio"]
        console.print(f"[bold green]截面因子多空组合回测[/bold green] {name}  "
                      f"标的数={res['n_symbols']} 截面数={res['n_dates']}")
        btab = Table(title=f"多空组合 {name}")
        btab.add_column("指标"); btab.add_column("值")
        for k in ("total_return", "annual_return", "sharpe", "sortino", "max_drawdown", "calmar"):
            btab.add_row(k, str(p.get(k)))
        console.print(btab)


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


async def _backtest(symbol, exchange, strategy, mode, gateway, years, exclude_limit=False, limit_pct=0.10, cost=False, leg2=None) -> None:
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
    res = run_strategy(mode, strat_class, vt, setting, bars, ee, sizes, gateway_name=gateway, cost=cost)
    await ee.stop()
    console.print(f"[bold]{mode.upper()} 结果 ({strategy})[/bold]"
                  + (" [cyan]含真实成本模型[/cyan]" if cost else ""))
    console.print(res)
    if mode == "live":
        console.print(f"[green]已切换至实盘路线（{gateway} 网关桩），委托已路由[/green]")


async def _wf(symbol, exchange, strategy, years, train_window, test_window, step, cost) -> None:
    from .backtest import walk_forward
    bars = await _fetch(symbol, exchange, years)
    if not bars:
        console.print("[red]无数据[/red]"); return
    if len(bars) < train_window + test_window:
        console.print(f"[red]样本不足：需 ≥ {train_window + test_window} 根，仅 {len(bars)}[/red]"); return
    vt = f"{symbol}.{exchange.upper()}"
    sizes = {vt: default_size(vt)}
    strat_class = {"multifactor": MultiFactorStrategy, "dual_ma": DualMaStrategy,
                   "vol_target": VolTargetStrategy, "pair": PairTradingStrategy}.get(strategy, MultiFactorStrategy)
    setting = {"size": sizes[vt], "max_pos": 1.0}
    res = walk_forward(bars, strat_class, setting, vt,
                       train_window=train_window, test_window=test_window, step=step,
                       sizes=sizes, cost=cost)
    agg = res.aggregate
    console.print(f"[bold]Walk-forward 验证 ({strategy})[/bold]"
                  + (" [cyan]含真实成本模型[/cyan]" if cost else ""))
    console.print(f"折数={agg['n_folds']}  训练窗={agg['train_window']}  测试窗={agg['test_window']}  步长={agg['step']}")
    console.print(f"样本外均值 Sharpe={agg['mean_sharpe']}  收益={agg['mean_total_return']}  "
                  f"收益波动={agg['std_total_return']}  盈利折占比={agg['positive_rate']}")
    table = Table(title="各折样本外绩效")
    table.add_column("折"); table.add_column("区间"); table.add_column("Sharpe"); table.add_column("收益")
    for f in res.folds:
        d = f.to_dict()
        table.add_row(str(d["fold"]), f"{d['start'][:10]}~{d['end'][:10]}",
                      str(d["sharpe"]), str(d["total_return"]))
    console.print(table)
    flag = "[red]⚠ 疑似过拟合[/red]" if res.overfit_suspected else "[green]样本外稳定[/green]"
    console.print(f"过拟合判定：{flag}  （全样本 Sharpe={res.detail.get('train_sharpe')}, "
                  f"样本外均值 Sharpe={res.detail.get('test_sharpe')}）")


async def _research(idea, asset_class) -> None:
    agent = ResearchAgent()
    out = await agent.research(idea, asset_class)
    console.print(f"[bold]研究：{idea}[/bold]")
    console.print(f"资产类别: {out.spec.asset_class}")
    console.print(f"假设: {out.spec.hypothesis}")
    console.print(f"候选因子: {[f.name for f in out.factors]}")
    console.print(f"风险要点: {out.spec.risk_notes}")
    console.print(f"生成策略代码安全: {out.code_safe}  错误: {out.code_errors}")


async def _seat(symbol: str, exchange: str, root: str) -> None:
    s = get_settings()
    root = root or s.seat_data_root or (
        f"{s.local_data_root}/qihuo/database/positioning" if s.local_data_root else ""
    )
    if not root:
        console.print("[red]未配置席位数据根目录[/red]（设置 QM_SEAT_DATA_ROOT 或 --root）")
        return
    try:
        from .research.factors.seat_futures import (
            compute_seat_factors, seat_df_from_tradingagents,
        )
        seat_df, total_oi = seat_df_from_tradingagents(root, symbol)
    except FileNotFoundError as e:
        console.print(f"[red]席位数据缺失[/red]: {e}")
        return
    if seat_df.empty:
        console.print("[red]该品种无席位数据[/red]")
        return
    console.print(f"[green]席位数据[/green]: {symbol} 共 {len(seat_df)} 交易日, "
                  f"{seat_df.shape[1]} 个席位（净持仓=多单-空单，按最活跃合约）")
    factors = compute_seat_factors(seat_df, total_oi, aggregate=True)

    table = Table(title=f"期货席位因子 {symbol} (F1–F8)")
    table.add_column("因子"); table.add_column("末值"); table.add_column("均值"); table.add_column("标准差")
    for name, ser in factors.items():
        table.add_row(name, f"{ser.iloc[-1]:.2f}", f"{ser.mean():.2f}", f"{ser.std():.2f}")
    console.print(table)

    # 尝试关联本地期货价格，计算各因子与次日收益的 IC（spearman）
    try:
        bars = await _fetch(f"{symbol.lower()}0", exchange, 1)
        from .research.factors.base import bars_to_df
        pdf = bars_to_df(bars)
        pdf["date"] = pd.to_datetime(pdf["datetime"]).dt.tz_localize(None).dt.normalize()
        pdf = pdf.sort_values("date")
        pdf["fwd_ret"] = pdf["close"].pct_change().shift(-1)
        ret_map = pdf.dropna(subset=["fwd_ret"]).set_index("date")["fwd_ret"]
        rows = []
        for name, ser in factors.items():
            sidx = pd.to_datetime(ser.index).normalize()
            aligned = ser.set_axis(sidx)
            common = aligned.index.intersection(ret_map.index)
            if len(common) >= 10:
                ic = aligned.loc[common].corr(ret_map.loc[common], method="spearman")
            else:
                ic = float("nan")
            rows.append((name, ic))
        ic_table = Table(title=f"{symbol} 席位因子 IC（vs 次日收益，spearman）")
        ic_table.add_column("因子"); ic_table.add_column("IC")
        for name, ic in rows:
            ic_table.add_row(name, f"{ic:.4f}" if ic == ic else "n/a")
        console.print(ic_table)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]未关联价格（无本地期货数据），跳过 IC：{str(exc)[:50]}[/yellow]")


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


async def _risk_report(profile, symbol, exchange, volume, price, equity) -> None:
    from datetime import datetime, timezone

    from .core.constant import Direction, Exchange as Ex, Offset
    from .core.gateway import OrderRequest
    from .risk import RiskEngine, RiskLimits
    from .risk.calendar import beijing_time, is_trading_time

    UTC = timezone.utc
    profile = (profile or "default").lower()
    if profile == "conservative":
        limits = RiskLimits.conservative()
    elif profile == "unlimited":
        limits = RiskLimits.unlimited()
    else:
        limits = RiskLimits()

    engine = RiskEngine(limits, initial_equity=equity)
    now = datetime.now(UTC)
    bj = beijing_time(now)
    vt = f"{symbol}.{exchange.upper()}"

    console.print("[bold cyan]==== 风控体检 ====[/bold cyan]")
    console.print(f"档位: [yellow]{profile}[/yellow]  合约: [yellow]{vt}[/yellow]  "
                  f"权益: ¥{equity:,.0f}  示例委托: {volume}手 @ {price}")

    # 限额档
    ltab = Table(title="限额档")
    ltab.add_column("项"); ltab.add_column("值")
    d = limits.to_dict()
    for k in ("max_order_volume", "max_position_volume", "max_margin_ratio",
              "max_daily_loss_ratio", "max_drawdown_ratio", "max_orders_per_day",
              "max_orders_per_minute", "check_trading_session", "allow_open",
              "self_trade_guard", "forbidden_symbols", "allowed_symbols"):
        ltab.add_row(k, str(d.get(k)))
    console.print(ltab)

    # 交易时段（实时）
    trading = is_trading_time(now, symbol, exchange)
    sess = engine.calendar.session_name(now, symbol, exchange)
    console.print(f"[bold]当前交易时段[/bold]: 北京时间 {bj:%Y-%m-%d %H:%M:%S}  "
                  f"交易日={engine.calendar.is_trading_day(bj.date())}  "
                  f"交易中={trading}  时段={sess}  "
                  f"今夜夜盘={engine.calendar.has_night_session(bj.date())}")

    # 示例委托：开仓
    open_req = OrderRequest(symbol=symbol, exchange=Ex(exchange.upper()),
                            direction=Direction.LONG, offset=Offset.OPEN,
                            volume=volume, price=price)
    open_dec = engine.check_order(open_req, equity=equity, now=now)
    _print_decision("开仓示例", open_dec)
    if not open_dec.passed and open_dec.code.value == "NOT_TRADING_TIME":
        console.print("[dim]提示：开仓被拒仅因当前非交易时段（时段闸门生效），并非配置错误。[/dim]")

    # 示例委托：平仓（无持仓 → 演示「平仓超持仓」守卫）
    close_req = OrderRequest(symbol=symbol, exchange=Ex(exchange.upper()),
                             direction=Direction.SHORT, offset=Offset.CLOSE,
                             volume=volume, price=price)
    close_dec = engine.check_order(close_req, equity=equity, now=now)
    _print_decision("平仓示例(无持仓)", close_dec)


def _print_decision(label: str, decision) -> None:
    if decision.passed:
        console.print(f"[green]✓ {label} 放行[/green] ({decision.code.value})")
    else:
        console.print(f"[red]✗ {label} 拒单[/red] {decision.code.value}: {decision.reason}")


if __name__ == "__main__":
    app()
