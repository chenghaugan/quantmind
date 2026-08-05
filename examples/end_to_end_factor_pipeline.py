# -*- coding: utf-8 -*-
"""端到端因子挖掘流水线示例（数据 → 挖掘 → 评测 → 去冗余 → 组合 → OOS 回测）。

这是把 QuantMind 研究链路的一条**完整可运行**示例：从合成面板出发，跑通
``run_pipeline``（真实 LLM 驱动，未配置 key 时自动降级为离线 Mock），再对
代表因子做一次 ``composite_backtest`` 组合权重优化与复合 alpha 回测，最后
打印结构化报告。

运行方式（在项目根 ``quantmind`` 目录下）：:

    .venv\\Scripts\\python.exe examples\\end_to_end_factor_pipeline.py --n-dates 120 --seeds 3

默认使用**离线 Mock provider**（无网络、无 key 也能跑）。若已在「设置」页或
``.env`` 配置了 ``QM_LLM_API_KEY``，脚本会自动切换到**真实 LLM** 做
co/ea/tot 迭代挖掘。用 ``--provider real|mock`` 强制指定。

参数：:
    --n-dates      合成面板日期数（默认 160）
    --n-symbols    合成面板标数（默认 12）
    --seeds        每个 seed 的搜索迭代轮数（默认 3，真实 LLM 建议 1~2 以省时）
    --algo         搜索算法 co | ea | tot（默认 co）
    --provider     real | mock | auto（默认 auto：有 key 用真实，否则 mock）
    --run-judge    是否用 LLM 对候选因子打分（默认关闭）
    --no-composite 跳过组合权重优化演示（默认开启）
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 保证在任意 cwd 下都能 import 到项目包（允许 examples/ 内直接运行）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)  # 与 SettingsService 的 .env 读取一致

import numpy as np
import pandas as pd

# 统一 stdout 为 UTF-8，避免 Windows GBK 控制台乱码
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import warnings
# numpy 相关矩阵在含 NaN/零方差列时会产生 benign RuntimeWarning，默认静默
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

from quantmind.research.factors.alpha_cs import Panel
from quantmind.research.pipeline import PipelineConfig, run_pipeline
from quantmind.research.combine import composite_backtest
from quantmind.ai.provider import LLMProvider


# --------------------------------------------------------------------------- #
# 1) 合成面板（确定性、可复现）：带趋势/反转/波动形态，使因子可见 IC
# --------------------------------------------------------------------------- #
def build_synthetic_panel(n_dates: int, n_symbols: int, seed: int = 0) -> Panel:
    """构造一个含动量/反转/波动形态的合成面板，供链路演示。

    每个标的有独立的漂移 + 随机游走 + 周期波动，保证 ``delta`` / ``ts_zscore`` /
    ``rank`` 类因子能测出非零 IC；同时注入跨截面差异使截面排序有信息。
    """
    idx = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    cols = [f"S{i}" for i in range(n_symbols)]
    rng = np.random.default_rng(seed)

    drift = rng.uniform(-0.0005, 0.0015, n_symbols)          # 每股长期漂移
    vol = rng.uniform(0.005, 0.03, n_symbols)                  # 每股波动率
    period = rng.integers(20, 60, n_symbols)                   # 每股周期（动量窗口）
    t = np.arange(n_dates)

    closes = np.zeros((n_dates, n_symbols))
    for j in range(n_symbols):
        phase = rng.uniform(0, 2 * np.pi)
        # 漂移 + 周期动量（过去 period 内涨跌 → 未来延续） + 白噪声
        trend = drift[j] * t
        cyc = 0.02 * np.sin(2 * np.pi * t / period[j] + phase)
        noise = rng.standard_normal(n_dates) * vol[j]
        # 用「累积收益」生成收盘价：既有趋势又有可持续的短期动量
        ret = np.concatenate([[0.0], np.diff(trend + cyc)]) + noise
        closes[:, j] = 100.0 * np.exp(np.cumsum(ret))

    close = pd.DataFrame(closes, index=idx, columns=cols)
    volume = pd.DataFrame(
        np.abs(rng.standard_normal((n_dates, n_symbols))) * 1e5 + 1e4,
        index=idx, columns=cols)
    return Panel(
        close=close,
        open=close * (1 - rng.uniform(0, 0.005, (n_dates, n_symbols))),
        high=close * (1 + rng.uniform(0, 0.008, (n_dates, n_symbols))),
        low=close * (1 - rng.uniform(0, 0.008, (n_dates, n_symbols))),
        volume=volume,
        amount=volume * close,
    )


# --------------------------------------------------------------------------- #
# 2) provider：auto → 有真实 LLM 配置用真实，否则 mock
# --------------------------------------------------------------------------- #
def make_provider(mode: str) -> LLMProvider:
    """按 ``mode`` 构造 provider。

    - ``mock``：强制离线 Mock。
    - ``real``：强制真实 LLM（需已配置 QM_LLM_API_KEY，否则报错提示）。
    - ``auto``（默认）：用 ``SettingsService`` 读已配置的 LLM；无 key 自动 fallback
      到 Mock。
    """
    if mode == "mock":
        from quantmind.ai.provider import build_provider
        return build_provider(name="mock")
    if mode == "real":
        from quantmind.api.services.settings_service import SettingsService
        svc = SettingsService()
        provider = svc.rebuild_provider()
        if type(provider).__name__ == "MockProvider":
            raise SystemExit(
                "[real] 未检测到真实 LLM 配置。请在「设置」页或 .env 配置 "
                "QM_LLM_API_KEY,或用 --provider auto/mock。")
        return provider
    # auto
    from quantmind.api.services.settings_service import SettingsService
    try:
        svc = SettingsService()
        provider = svc.rebuild_provider()
        if type(provider).__name__ != "MockProvider":
            return provider
    except Exception:  # noqa: BLE001
        pass
    from quantmind.ai.provider import build_provider
    return build_provider(name="mock")


# --------------------------------------------------------------------------- #
# 3) 打印
# --------------------------------------------------------------------------- #
def _fmt(x, d=4):
    return "—" if x is None else f"{float(x):.{d}f}"


def print_report(report: dict) -> None:
    s = report["summary"]
    print("\n================ 流水线汇总 ================")
    print(f"算法={s['algo'].upper()}  种子数={s['seed_count']}  "
          f"候选={s['candidate_count']}  代表={s['representative_count']}  "
          f"回测数={s['backtested_count']}")
    print(f"mean_train_ic={_fmt(s['mean_train_ic'])}  mean_val_ic={_fmt(s['mean_val_ic'])}  "
          f"mean_test_ic={_fmt(s['mean_test_ic'])}  mean_test_sharpe={_fmt(s['mean_test_sharpe'])}")
    print("\n---------------- 代表因子（去冗余后） ----------------")
    for st in report["steps"]:
        red = f" [吸收{len(st['removed_redundant'])}冗余]" if st["removed_redundant"] else ""
        print(f"  {st['expression'][:46]:46} train_ic={_fmt(st['train_ic'])} "
              f"val_ic={_fmt(st['val_ic'])} OOS_sharpe={_fmt(st['test_sharpe'])}{red}")
    if report.get("composite"):
        c = report["composite"]
        pf = c["portfolio"]
        print("\n---------------- 复合 alpha 组合（权重优化） ----------------")
        w = "  ".join(f"{k[:22]}:{v:.3f}" for k, v in c["weights"].items())
        print(f"  方案={c['scheme']}  权重: {w}")
        print(f"  复合 OOS Sharpe={_fmt(pf.get('sharpe'))}  "
              f"total_return={_fmt(pf.get('total_return'))}  "
              f"max_drawdown={_fmt(pf.get('max_drawdown'))}  "
              f"fwd_IC={_fmt((c['ic_report'] or {}).get('ic_mean'))}")


# --------------------------------------------------------------------------- #
# 4) 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="端到端因子挖掘流水线示例")
    ap.add_argument("--n-dates", type=int, default=160)
    ap.add_argument("--n-symbols", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=3, help="每 seed 搜索轮数")
    ap.add_argument("--algo", default="co", choices=["co", "ea", "tot"])
    ap.add_argument("--provider", default="auto", choices=["auto", "real", "mock"])
    ap.add_argument("--run-judge", action="store_true", help="用 LLM 对候选打分")
    ap.add_argument("--no-composite", action="store_true", help="跳过组合演示")
    ap.add_argument("--rounds", type=int, default=0,
                    help="覆盖搜索迭代轮数（默认取 --seeds 值）")
    args = ap.parse_args()

    print("== 1/5 构建面板 ==")
    panel = build_synthetic_panel(args.n_dates, args.n_symbols)
    print(f"  面板: {len(panel.dates)} 日 × {len(panel.symbols)} 标的")

    print("\n== 2/5 构造 provider ==")
    provider = make_provider(args.provider)
    print(f"  provider={provider.name}（{type(provider).__name__}）")

    seeds = ["delta(close,5)",
             "ts_zscore(close,20)",
             "rank(close,10)",
             "ts_rank(close,20)",
             "corr(close,volume,10)"]
    print(f"  种子: {len(seeds)} 条；每 seed 搜索轮数={args.rounds or args.seeds}")

    print("\n== 3/5 端到端流水线（挖掘→去冗余→逐因子OOS回测→复合组合） ==")
    cfg = PipelineConfig(
        seeds=seeds,
        algo=args.algo,
        rounds=args.rounds or args.seeds,
        forward_periods=1,
        train_frac=0.5,
        val_frac=0.2,
        dedup_threshold=0.7,
        run_judge=args.run_judge,
        max_candidates=8,
        persist_pairs=False,
        run_composite=not args.no_composite,
        composite_scheme="icir",
    )
    report = run_pipeline(panel, config=cfg, provider=provider)

    print("\n== 4/5 独立组合权重优化演示（composite_backtest） ==")
    if not args.no_composite:
        try:
            bt = composite_backtest(
                [s["expression"] for s in report["steps"]],
                panel,
                scheme="min_var",
                forward_periods=1,
            )
            print(f"  min_var 方案权重: {bt['weights']}")
            print(f"  复合 Sharpe={_fmt(bt['portfolio'].get('sharpe'))}  "
                  f"total_return={_fmt(bt['portfolio'].get('total_return'))}")
        except Exception as exc:  # noqa: BLE001
            print(f"  （组合回测跳过: {exc}）")

    print("\n== 5/5 报告 ==")
    print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
