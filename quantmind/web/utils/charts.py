"""Plotly 图表组件（统一 QuantMind 深色模板）。

所有图表共用 :data:`QM_TEMPLATE` 模板，保证与 Streamlit 主题一致；
输入为空或缺列时返回「空状态」图而不是抛异常。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .theme import COLORS, PLOTLY_COLORWAY

QM_TEMPLATE = "quantmind"

# --------------------------------------------------------------------------
# 注册统一模板
# --------------------------------------------------------------------------
if QM_TEMPLATE not in pio.templates:
    _tpl = go.layout.Template()
    _tpl.layout = go.Layout(
        colorway=PLOTLY_COLORWAY,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text_muted"], size=12,
                  family='Inter, "PingFang SC", "Microsoft YaHei", sans-serif'),
        title=dict(font=dict(color=COLORS["text"], size=14), x=0.01, xanchor="left"),
        margin=dict(l=48, r=24, t=44, b=40),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=COLORS["surface_alt"], bordercolor=COLORS["border"],
                        font=dict(color=COLORS["text"], size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis=dict(gridcolor=COLORS["border_soft"], zerolinecolor=COLORS["border"],
                   linecolor=COLORS["border"], showspikes=False),
        yaxis=dict(gridcolor=COLORS["border_soft"], zerolinecolor=COLORS["border"],
                   linecolor=COLORS["border"]),
    )
    pio.templates[QM_TEMPLATE] = _tpl


def _base_layout(fig: go.Figure, title: str = "", height: int = 400, **kw) -> go.Figure:
    fig.update_layout(template=QM_TEMPLATE, title=title, height=height, **kw)
    return fig


def empty_figure(msg: str = "暂无数据", height: int = 300) -> go.Figure:
    """空状态占位图。"""
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False,
                       font=dict(size=14, color=COLORS["text_dim"]),
                       xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base_layout(fig, height=height)


def _to_df(rows, required: Sequence[str]) -> Optional[pd.DataFrame]:
    if not rows:
        return None
    try:
        df = pd.DataFrame(rows)
    except Exception:  # noqa: BLE001
        return None
    if df.empty or any(c not in df.columns for c in required):
        return None
    return df


# --------------------------------------------------------------------------
# 行情
# --------------------------------------------------------------------------
def create_price_chart(bars: list, title: str = "", ma: Sequence[int] = (5, 20, 60),
                       height: int = 560) -> go.Figure:
    """K 线 + 均线 + 成交量（红涨绿跌）。"""
    df = _to_df(bars, ["datetime", "open", "high", "low", "close"])
    if df is None:
        return empty_figure("无行情数据", height=height)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    has_vol = "volume" in df.columns

    fig = make_subplots(
        rows=2 if has_vol else 1, cols=1, shared_xaxes=True,
        vertical_spacing=0.04, row_heights=[0.74, 0.26] if has_vol else [1.0],
    )

    fig.add_trace(
        go.Candlestick(
            x=df["datetime"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="K线",
            increasing=dict(line=dict(color=COLORS["up"], width=1), fillcolor=COLORS["up"]),
            decreasing=dict(line=dict(color=COLORS["down"], width=1), fillcolor=COLORS["down"]),
        ),
        row=1, col=1,
    )

    ma_colors = ["#f59e0b", "#60a5fa", "#a78bfa", "#22d3ee"]
    for i, w in enumerate(ma or []):
        if len(df) > w:
            fig.add_trace(
                go.Scatter(x=df["datetime"], y=df["close"].rolling(w).mean(),
                           name=f"MA{w}", mode="lines",
                           line=dict(color=ma_colors[i % len(ma_colors)], width=1.3)),
                row=1, col=1,
            )

    if has_vol:
        colors = [COLORS["up"] if c >= o else COLORS["down"]
                  for c, o in zip(df["close"], df["open"])]
        fig.add_trace(
            go.Bar(x=df["datetime"], y=df["volume"], name="成交量",
                   marker_color=colors, marker_line_width=0, opacity=0.75),
            row=2, col=1,
        )
        fig.update_yaxes(title_text="量", row=2, col=1)

    fig.update_yaxes(title_text="价格", row=1, col=1)
    return _base_layout(fig, title, height, xaxis_rangeslider_visible=False,
                        showlegend=True, bargap=0.15)


# --------------------------------------------------------------------------
# 回测
# --------------------------------------------------------------------------
def create_equity_curve(equity_curve: list, title: str = "净值曲线",
                        benchmark: Optional[list] = None, height: int = 420) -> go.Figure:
    """净值曲线（自动归一化为 1.0 起点），带渐变填充与关键标注。"""
    df = _to_df(equity_curve, ["equity"])
    if df is None:
        return empty_figure("无权益数据", height=height)

    x = pd.to_datetime(df["date"]) if "date" in df.columns else pd.RangeIndex(len(df))
    init = float(df["equity"].iloc[0]) or 1.0
    nav = (df["equity"].astype(float) / init).round(6)

    fig = go.Figure()

    # 策略净值 — 渐变填充
    fig.add_trace(go.Scatter(
        x=x, y=nav, mode="lines", name="策略净值",
        line=dict(color=COLORS["primary"], width=2.4, shape="spline", smoothing=0.8),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,.08)",
        hovertemplate="日期: %%{x|%%Y-%%m-%%d}<br>净值: %%{y:.4f}<extra></extra>",
    ))

    if benchmark:
        bdf = _to_df(benchmark, ["equity"])
        if bdf is not None:
            bx = pd.to_datetime(bdf["date"]) if "date" in bdf.columns else pd.RangeIndex(len(bdf))
            binit = float(bdf["equity"].iloc[0]) or 1.0
            bnav = (bdf["equity"].astype(float) / binit).round(6)
            fig.add_trace(go.Scatter(
                x=bx, y=bnav, mode="lines", name="基准",
                line=dict(color=COLORS["text_dim"], width=1.6, dash="dot"),
                hovertemplate="基准: %%{y:.4f}<extra></extra>",
            ))

    # 起点参考线
    fig.add_hline(y=1.0, line=dict(color=COLORS["border"], width=1, dash="dash"))

    # 标注最高点与最终点
    peak_idx = nav.idxmax()
    peak_x = x[peak_idx] if not isinstance(x, pd.RangeIndex) else peak_idx
    peak_y = nav.iloc[peak_idx]
    fig.add_trace(go.Scatter(
        x=[peak_x], y=[peak_y], mode="markers+text",
        marker=dict(size=8, color=COLORS["up"], symbol="diamond",
                    line=dict(color="white", width=1.2)),
        text=[f"峰值 {peak_y:.2f}"], textposition="top center",
        textfont=dict(size=10, color=COLORS["text"]),
        showlegend=False, hoverinfo="skip",
    ))

    final_x = x.iloc[-1] if not isinstance(x, pd.RangeIndex) else len(nav) - 1
    final_y = nav.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[final_x], y=[final_y], mode="markers+text",
        marker=dict(size=8, color=COLORS["primary"], symbol="circle",
                    line=dict(color="white", width=1.2)),
        text=[f"终值 {final_y:.2f}"], textposition="top right",
        textfont=dict(size=10, color=COLORS["text"]),
        showlegend=False, hoverinfo="skip",
    ))

    return _base_layout(fig, title, height, yaxis_title="净值", showlegend=True,
                        yaxis=dict(tickformat=".2f", gridcolor=COLORS["border_soft"]))


def create_drawdown_chart(equity_curve: list, title: str = "回撤",
                          height: int = 280) -> go.Figure:
    """水下回撤曲线，标注最大回撤区间。"""
    df = _to_df(equity_curve, ["equity"])
    if df is None:
        return empty_figure("无权益数据", height=height)

    x = pd.to_datetime(df["date"]) if "date" in df.columns else pd.RangeIndex(len(df))
    eq = df["equity"].astype(float)
    running_max = eq.cummax()
    dd = eq / running_max - 1.0

    fig = go.Figure()

    # 回撤填充区域 — 更柔和的渐变
    fig.add_trace(go.Scatter(
        x=x, y=dd, fill="tozeroy", name="回撤",
        line=dict(color=COLORS["danger"], width=1.6, shape="spline", smoothing=0.6),
        fillcolor="rgba(242,72,62,.15)",
        hovertemplate="回撤: %{y:.2%}<extra></extra>",
    ))

    # 零线 — 加粗高亮
    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1.2, dash="solid"))

    # 标注最大回撤点
    trough_idx = dd.idxmin()
    if dd.iloc[trough_idx] < -0.01:  # 只在回撤超过1%时标注
        trough_x = x[trough_idx] if not isinstance(x, pd.RangeIndex) else trough_idx
        trough_y = dd.iloc[trough_idx]
        fig.add_trace(go.Scatter(
            x=[trough_x], y=[trough_y], mode="markers+text",
            marker=dict(size=9, color=COLORS["danger"], symbol="diamond",
                        line=dict(color="white", width=1.5)),
            text=[f"最大回撤 {trough_y:.1%}"], textposition="bottom center",
            textfont=dict(size=11, color=COLORS["text"]),
            showlegend=False, hoverinfo="skip",
        ))

    return _base_layout(fig, title, height, yaxis_title="回撤",
                        yaxis_tickformat=".1%", showlegend=False,
                        yaxis=dict(gridcolor=COLORS["border_soft"],
                                   zerolinecolor=COLORS["text_muted"],
                                   zerolinewidth=1.2))


def create_returns_histogram(returns: Sequence[float], title: str = "收益分布",
                             height: int = 260) -> go.Figure:
    """日收益分布直方图。"""
    vals = [float(r) for r in (returns or []) if r is not None]
    if not vals:
        return empty_figure("无收益序列", height=height)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=vals, nbinsx=40, marker_color=COLORS["violet"],
                               opacity=0.85, name="日收益"))
    fig.add_vline(x=0, line=dict(color=COLORS["border"], width=1, dash="dash"))
    return _base_layout(fig, title, height, showlegend=False,
                        xaxis_tickformat=".2%", hovermode="closest")


def create_monthly_heatmap(equity_curve: list, title: str = "月度收益",
                           height: int = 300) -> go.Figure:
    """按年/月聚合的收益热力图（红涨绿跌）。"""
    df = _to_df(equity_curve, ["date", "equity"])
    if df is None:
        return empty_figure("无权益数据", height=height)
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")["equity"].astype(float)
    monthly = s.resample("ME").last().pct_change().dropna()
    if monthly.empty:
        return empty_figure("样本不足一个月", height=height)

    pivot = pd.DataFrame({
        "year": monthly.index.year, "month": monthly.index.month, "ret": monthly.values,
    }).pivot(index="year", columns="month", values="ret")

    fig = go.Figure(go.Heatmap(
        z=pivot.values * 100,
        x=[f"{m}月" for m in pivot.columns],
        y=[str(y) for y in pivot.index],
        colorscale=[[0, COLORS["down"]], [0.5, "#111b2e"], [1, COLORS["up"]]],
        zmid=0, texttemplate="%{z:.1f}", textfont=dict(size=10),
        colorbar=dict(title="%", thickness=10),
    ))
    return _base_layout(fig, title, height, hovermode="closest")


# --------------------------------------------------------------------------
# 因子
# --------------------------------------------------------------------------
def create_ic_chart(ic_decay: list, title: str = "IC 衰减", height: int = 280) -> go.Figure:
    """IC 衰减柱状图（正 IC 红、负 IC 绿）。"""
    vals = [v for v in (ic_decay or []) if v is not None]
    if not vals:
        return empty_figure("无 IC 衰减数据", height=height)
    periods = [f"T+{i}" for i in range(1, len(vals) + 1)]
    colors = [COLORS["up"] if v >= 0 else COLORS["down"] for v in vals]
    fig = go.Figure(go.Bar(x=periods, y=vals, marker_color=colors,
                           marker_line_width=0, name="IC",
                           text=[f"{v:.3f}" for v in vals], textposition="outside",
                           textfont=dict(size=10)))
    fig.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
    return _base_layout(fig, title, height, yaxis_title="IC",
                        showlegend=False, hovermode="closest")


def create_quantile_bar(returns: Dict[str, float], title: str = "分位收益",
                        height: int = 280) -> go.Figure:
    """分位组收益柱状图。"""
    if not returns:
        return empty_figure("无分位收益数据", height=height)
    keys = list(returns.keys())
    vals = [returns[k] for k in keys]
    colors = [COLORS["up"] if v >= 0 else COLORS["down"] for v in vals]
    fig = go.Figure(go.Bar(x=keys, y=vals, marker_color=colors, marker_line_width=0))
    fig.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
    return _base_layout(fig, title, height, yaxis_tickformat=".2%",
                        showlegend=False, hovermode="closest")


def create_factor_radar(scores: Dict[str, float], title: str = "因子画像",
                        height: int = 320) -> go.Figure:
    """因子多维评分雷达图（各维度需已归一化到 0~1）。"""
    if not scores:
        return empty_figure("无评分数据", height=height)
    labels = list(scores.keys())
    vals = [max(0.0, min(1.0, float(scores[k]))) for k in labels]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=labels + [labels[0]], fill="toself",
        line=dict(color=COLORS["primary"], width=2),
        fillcolor="rgba(59,130,246,.22)", name="评分",
    ))
    fig.update_polars(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(visible=True, range=[0, 1], gridcolor=COLORS["border_soft"],
                        tickfont=dict(size=9)),
        angularaxis=dict(gridcolor=COLORS["border_soft"], tickfont=dict(size=10)),
    )
    return _base_layout(fig, title, height, showlegend=False, hovermode="closest")


# --------------------------------------------------------------------------
# Walk-Forward / 优化
# --------------------------------------------------------------------------
def create_fold_chart(folds: List[dict], metric: str = "sharpe",
                      title: str = "各折表现", height: int = 300) -> go.Figure:
    """Walk-Forward 每折指标柱状图。"""
    if not folds:
        return empty_figure("无分折数据", height=height)
    x = [f"F{f.get('fold', i)}" for i, f in enumerate(folds)]
    y = [f.get(metric) or 0 for f in folds]
    colors = [COLORS["up"] if v >= 0 else COLORS["down"] for v in y]
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=colors, marker_line_width=0,
                           text=[f"{v:.2f}" for v in y], textposition="outside",
                           textfont=dict(size=10)))
    fig.add_hline(y=0, line=dict(color=COLORS["border"], width=1))
    return _base_layout(fig, title, height, yaxis_title=metric,
                        showlegend=False, hovermode="closest")


def create_optimize_scatter(results: List[dict], metric: str = "sharpe",
                            title: str = "参数寻优结果", height: int = 340) -> go.Figure:
    """参数组合 -> 指标散点图（按指标着色）。"""
    if not results:
        return empty_figure("无寻优结果", height=height)
    labels = [", ".join(f"{k}={v}" for k, v in (r.get("setting") or {}).items())
              for r in results]
    y = [r.get(metric, r.get("sharpe")) or 0 for r in results]
    fig = go.Figure(go.Scatter(
        x=list(range(len(y))), y=y, mode="markers", text=labels,
        hovertemplate="%{text}<br>" + metric + "=%{y:.4f}<extra></extra>",
        marker=dict(size=9, color=y, colorscale="Viridis", showscale=True,
                    colorbar=dict(thickness=10), line=dict(width=0)),
    ))
    best = max(range(len(y)), key=lambda i: y[i])
    fig.add_trace(go.Scatter(x=[best], y=[y[best]], mode="markers", name="最优",
                             marker=dict(size=16, color=COLORS["amber"],
                                         symbol="star", line=dict(width=0))))
    return _base_layout(fig, title, height, xaxis_title="参数组合序号",
                        yaxis_title=metric, hovermode="closest", showlegend=False)


def create_gauge(value: float, title: str, vmin: float = -1.0, vmax: float = 3.0,
                 good: float = 1.0, height: int = 220) -> go.Figure:
    """绩效仪表盘 — 精致版，带渐变阈值色阶与状态标签。"""
    v = 0.0 if value is None or value != value else float(value)

    # 根据值动态选择颜色与状态
    if v >= good:
        color = COLORS["up"]
        status_text = "✓ 达标"
    elif v >= 0:
        color = COLORS["amber"]
        status_text = "△ 待改进"
    else:
        color = COLORS["down"]
        status_text = "✗ 警示"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=v,
        number=dict(
            font=dict(size=30, color=COLORS["text"], family="Inter, sans-serif"),
            valueformat=".2f",
        ),
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:12px;color:{color}'>{status_text}</span>",
            font=dict(size=13, color=COLORS["text_muted"]),
        ),
        gauge=dict(
            axis=dict(
                range=[vmin, vmax],
                tickwidth=1.5,
                tickcolor=COLORS["text_muted"],
                tickfont=dict(size=10, color=COLORS["text_muted"]),
            ),
            bar=dict(
                color=color,
                thickness=0.75,
                line=dict(color="rgba(255,255,255,0.25)", width=2),
            ),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            bordercolor="rgba(0,0,0,0)",
            steps=[
                dict(range=[vmin, 0], color="rgba(242,72,62,.20)"),
                dict(range=[0, good], color="rgba(245,158,11,.16)"),
                dict(range=[good, vmax], color="rgba(18,184,134,.20)"),
            ],
            threshold=dict(
                line=dict(color=color, width=4),
                thickness=0.85,
                value=v,
            ),
        ),
    ))

    return _base_layout(fig, "", height, margin=dict(l=28, r=28, t=56, b=14))


def create_multi_line(series: Dict[str, Sequence[float]], x: Optional[Sequence] = None,
                      title: str = "", height: int = 340, yaxis_title: str = "") -> go.Figure:
    """多序列折线图。"""
    series = {k: v for k, v in (series or {}).items() if v is not None and len(v)}
    if not series:
        return empty_figure("无数据", height=height)
    fig = go.Figure()
    for i, (name, ys) in enumerate(series.items()):
        xs = x if x is not None else list(range(len(ys)))
        fig.add_trace(go.Scatter(x=xs, y=list(ys), mode="lines", name=name,
                                 line=dict(width=1.8,
                                           color=PLOTLY_COLORWAY[i % len(PLOTLY_COLORWAY)])))
    return _base_layout(fig, title, height, yaxis_title=yaxis_title, showlegend=True)


def create_event_timeline(counts: Dict[str, int], title: str = "事件分布",
                          height: int = 260) -> go.Figure:
    """事件类型计数条形图。"""
    if not counts:
        return empty_figure("尚无事件", height=height)
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    fig = go.Figure(go.Bar(
        x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
        marker=dict(color=[PLOTLY_COLORWAY[i % len(PLOTLY_COLORWAY)]
                           for i in range(len(items))]),
        text=[v for _, v in items], textposition="outside", textfont=dict(size=10),
    ))
    return _base_layout(fig, title, height, showlegend=False, hovermode="closest",
                        margin=dict(l=90, r=30, t=44, b=30))
