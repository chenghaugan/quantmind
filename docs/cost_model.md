# 结构化交易成本模型

> 回测若不计入真实交易成本，Sharpe / 收益会被严重高估。本模块把"零成本"假设换成
> 按品种差异化的真实成本模型。

## 为什么需要

原回测引擎只用**单一固定万 2 费率 + 固定点数滑点**，且 `run_strategy` 的 backtest
分支根本没有把 `commission`/`slippage` 透传给引擎（一直走默认万 2、零滑点）。
真实市场的成本结构远比这复杂：

- **期货平今（close-today）手续费与开仓差异巨大**：股指期货 2019 年后平今免收，
  此前曾高达开仓费率的数十倍；螺纹/热卷等商品平今也常免收。
- **A 股卖出收千分之一印花税**，且单笔有最低 5 元手续费门槛。
- **滑点**按最小变动价位（tick）倍数建模比固定点数更贴近实盘。
- **保证金**占用影响资金利用率与最大可开仓（杠杆约束）。
- **冲击成本**在量大时对成交价有线性拖累。

## 成本参数（CostModel）

| 字段 | 含义 | 备注 |
|------|------|------|
| `commission_rate` | 按成交额比例手续费 | 如 RB 万 1 |
| `commission_per_lot` | 每手固定手续费 | 如国债 ≈3 元/手 |
| `min_commission` | 单笔最低手续费 | A 股 5 元 |
| `close_today_rate_multiplier` | 平今费率倍率（相对开仓） | 0 = 平今免费；股指设 0 |
| `close_yesterday_rate_multiplier` | 平昨费率倍率 | 多数 = 1 |
| `stamp_tax_rate` | 印花税（仅卖出） | A 股千 1 |
| `slippage_ticks` | 滑点 = ticks × tick_size | 贴近实盘 |
| `slippage_rate` | 滑点 = 开盘价 × 比例 | 替代固定点数 |
| `tick_size` | 最小变动价位 | 用于 tick 滑点 |
| `margin_rate` | 保证金率 | 记录/约束用 |
| `impact_rate` | 冲击成本比例 | price×vol×size×rate |

## 预设成本表

`CONTRACT_COST_TABLE` 覆盖国内主要品种（数值为**常见近似，需按交易所当期公告校准**）：

- 金融期货 CFFEX：IF/IC/IH 平今免（2019 后）；国债 TF/T/TS 按手 ≈3 元、低保证金。
- 上期所 SHFE：RB/HC 平今免；CU/AG/AU/RU/AL/ZN/NI 等按品种。
- 大商所 DCE / 郑商所 CZCE / 能源中心 INE / 广期所 GFEX：主要品种。
- A 股 SSE/SZSE：万 2.5 佣金 + 卖出千 1 印花税 + 最低 5 元。
- 港股 HKEX、期权 OPTION：近似。

解析顺序：`精确 symbol → 品种前缀（如 rb0→RB）→ 交易所默认 → 商品期货通用`。

## 平今判定

引擎维护**开仓批次记账**（`self._open_lots`，FIFO）。平仓时从最早批次弹出，
若被平批次的开仓日期与平仓日期相同（UTC 自然日，与中国交易日对齐），则计为平今，
该部分按 `close_today_rate_multiplier` 计费；同一笔成交可同时含平今与平昨，按比例拆分。

> 注：日频回测中由于"次根开盘成交"，开仓与平仓至少间隔一根 K 线，通常触发的是平昨；
> 平今主要在分钟级回测或同根反手时出现。这是真实行为，无需特殊处理。

## 保证金

保证金仅作**占用记录**（`margin_used`）与可选**容量约束**（`enforce_margin=True` 时
在下单前检查 `可用资金 = 余额 − 已冻结保证金`）。权益曲线公式 `equity = 余额 + 浮动盈亏`
不变（保证金是账户内部分配，不重复扣减），因此不影响绩效分析的现有逻辑。

## 用法

### CLI

```bash
# 启用真实成本模型
python -m quantmind.cli backtest --symbol rb0 --exchange SHFE --strategy multifactor --cost

# 配对 + 成本
python -m quantmind.cli backtest --symbol rb0 --exchange SHFE --strategy pair --leg2 hc0.SHFE --cost
```

### API

```json
POST /backtest
{
  "strategy": "multifactor",
  "symbol": "rb0",
  "exchange": "SHFE",
  "cost": true
}
```

### 代码

```python
from quantmind.backtest import BacktestEngine

# 默认（兼容旧式单一费率）
eng = BacktestEngine(data, commission=0.0002)

# 启用内置真实成本表
eng = BacktestEngine(data, cost_table=True)

# 自定义成本表
from quantmind.backtest import CostModel
eng = BacktestEngine(data, cost_table={"rb0.SHFE": CostModel(commission_rate=0.0001)})
```

## 报告字段

`PerformanceReport` 新增：`total_commission` / `total_stamp_tax` / `total_impact` /
`total_slippage` / `total_cost` / `margin_used` / `cost_ratio`（总成本 ÷ |净收益|）。

## 校准提醒

`CONTRACT_COST_TABLE` 中的数值是**示例性近似值**，交易所费率政策会随时间调整
（尤其股指期货平今规则在 2015/2019 年有过大幅变化）。长周期回测前请按**当期交易所
官方公告**校准对应品种的 `commission_rate` / `close_today_rate_multiplier` / `margin_rate`。
