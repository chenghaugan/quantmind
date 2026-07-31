# 回测严谨性：walk-forward 与截面因子多空回测

QuantMind 在 P0 阶段补齐了两块回测严谨性能力，避免「单段历史上的偶然好成绩被误认为
稳定 alpha」与「因子 IC 高却做不出组合」的脱节。

## 1. Walk-forward 滚动样本外验证

把一段历史切成多个「训练窗 + 测试窗」折，每折在**测试窗**上跑回测（训练窗仅作指标预热
burn-in，本框架策略无待拟合参数，故等价于滚动样本外验证），逐折收集绩效，并用
`diagnose_overfitting` 把「全样本(in-sample)」与「各折样本外均值」对比，给出过拟合预警。

模块：`quantmind/backtest/walkforward.py`，函数 `walk_forward(...)`。

```python
from quantmind.backtest import walk_forward
from quantmind.strategy.dual_ma import DualMaStrategy
from quantmind.core.contracts import default_size

res = walk_forward(
    bars, DualMaStrategy,
    {"fast": 5, "slow": 20, "size": default_size("rb0.SHFE"), "max_pos": 1.0},
    "rb0.SHFE",
    train_window=250, test_window=60, step=60, sizes={"rb0.SHFE": default_size("rb0.SHFE")},
    cost=True,   # 可选：启用真实成本模型
)
print(res.aggregate)          # n_folds / 均值Sharpe / 均值收益 / 收益波动 / 盈利折占比
print(res.overfit_suspected)  # True=疑似过拟合
for f in res.folds:
    print(f.to_dict())        # 每折 Sharpe/收益/区间
```

CLI 一键：

```bash
python -m quantmind.cli wf --symbol rb0 --exchange SHFE --strategy dual_ma \
    --years 3 --train-window 250 --test-window 60 --step 60 --cost
```

判定逻辑（见 `backtest/diagnostics.diagnose_overfitting`）：若样本外 Sharpe < 0.5×样本内
Sharpe，或样本内盈利而样本外亏损，判定为疑似过拟合。

## 2. 截面因子 → 多空组合回测（研究-回测闭环）

把 `research/factors/alpha_cs` 在面板上算出的**严格截面因子**，直接转成每日横截面排名
驱动的多空组合，用「次根」前向收益做样本外回测，得到可比较的权益曲线与绩效指标；同时
附上同一因子的截面 IC 报告。这样因子研究（IC）与组合表现（Sharpe/回撤）在同一条流水线上
闭环。

模块：`quantmind/research/cross_sectional_backtest.py`，函数 `cross_sectional_backtest(...)`。

```python
from quantmind.research.factors.alpha_cs import Panel
from quantmind.research.cross_sectional_backtest import cross_sectional_backtest

res = cross_sectional_backtest(panel, "alpha021", forward_periods=1, n_groups=5, long_short=True)
print(res["ic_report"])       # 该因子的截面 IC / 衰减 / 单调性 / 多空组合
print(res["portfolio"])       # 多空组合的 Sharpe / 收益 / 回撤 / 日收益序列
```

CLI 一键（`--bt` 在截面评估的同时跑组合回测）：

```bash
python -m quantmind.cli cs --symbols rb0,hc0,bu0,i0 --exchange SHFE --name alpha021 --bt
```

实现要点：

- **无前视**：第 t 日信号只用 t 日及之前数据（含 close[t]），组合收益用 close[t]→close[t+fp]
  前向收益，符合严谨回测约定。
- **分组自适应**：`n_groups` 上限自动取可用标的数（默认 5，但 4 标的环境会自动降为 4），
  保证小宇宙也能成组；头组做多、尾组做空（等权）。
- **成本近似**：`cost_rate` 直接按每期双边成本扣减组合收益（如 0.001）。
- 权益曲线经 `backtest/analyzer.PerformanceAnalyzer` 得标准化指标，与研究端 IC 报告一并输出。

> 注：以上均为原生实现，未复制任何第三方仓库代码；组合收益为简化的横截面等权多空，
> 不含个股权重优化、行业中性化、换手约束等进阶处理（可在 `cross_sectional_backtest`
> 基础上扩展）。
