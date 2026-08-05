"""Benchmark 命令行入口：评估 LLM 在因子挖掘各任务的得分矩阵。

用法（离线 Mock，无需 key）：

    python -m quantmind.benchmark.run_benchmark

需真实模型时配置 API key（见 :mod:`quantmind.ai.provider`），并传入 provider 名。
"""
from __future__ import annotations

import argparse
import asyncio

from .runner import BenchmarkConfig, run_benchmark, summarize_matrix


def _make_panel():
    """构造一个小的确定性演示面板（离线评测用）。"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta, timezone

    from quantmind.research.factors.alpha_cs import Panel

    rng = np.random.default_rng(1)
    dates = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(120)]
    cols = [f"S{i}" for i in range(8)]
    close = pd.DataFrame(np.abs(rng.normal(100, 10, (120, 8))), index=dates, columns=cols)
    return Panel(close=close, open=close * 0.99, high=close * 1.02,
                 low=close * 0.98, volume=pd.DataFrame(np.abs(rng.normal(1000, 100, (120, 8))),
                                                        index=dates, columns=cols))


async def _main() -> None:
    parser = argparse.ArgumentParser(description="QuantMind LLM 因子挖掘评测")
    parser.add_argument("--provider", default="mock", help="mock | 真实模型名（需 API key）")
    parser.add_argument("--label", default="", help="结果标签")
    parser.add_argument("--task", default="", help="只跑单个任务（可选）")
    args = parser.parse_args()

    from ..ai.provider import build_provider
    provider = build_provider(name=args.provider)
    label = args.label or args.provider

    panel = _make_panel()
    cfg = BenchmarkConfig()
    if args.task:
        cfg.tasks = [args.task]

    if args.task:
        from .tasks import run_task
        r = await run_task(args.task, provider, panel)
        print(r.to_dict())
        return

    res = await run_benchmark(provider, panel=panel, label=label, config=cfg)
    print(f"\n== Benchmark 矩阵: {label} ==")
    import json
    print(json.dumps(res.to_matrix_dict(), ensure_ascii=False, indent=2))
    print("\n== 按任务评分 ==")
    for t, scores in res.scores_by_task().items():
        print(f"  {t:14s} {scores}")


if __name__ == "__main__":
    asyncio.run(_main())
