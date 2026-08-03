# 实盘化（Live Trading）架构与上线指南

> 阶段目标：在不破坏现有回测 / 模拟 / Web 架构的前提下，补齐实盘化（P0 严谨性）
> 缺失的核心能力——**硬风控闸门、本地订单簿状态机、持仓/资金对账**——使
> 「切换路线即可跑实盘」从桩状态变为真正可用的安全链路。

---

## 1. 设计原则

- **风控是代码强制的闸门，不是策略的自觉。** 任何委托在到达网关前必须先过
  `RiskEngine.check_order()`；拒单返回 `RiskDecision(passed=False)` 并广播
  `EVENT_RISK`，**不抛异常打断策略循环**。策略无法说服风控放行，也无法用参数把硬阈值调没
  （阈值只能在引擎装配时传入）。
- **一套策略代码切路线。** `StrategyContext` + `run_strategy(mode)`：backtest / paper / live
  共用同一份策略，换 context 即切换路线。
- **熔断不因策略重启自动清除。** `resume()` 必须显式调用且留痕，杜绝「策略自己把风控关掉」。
- **减仓必须放行。** 限额只约束「增加风险敞口」的方向；已超限时减仓若被拒会永久锁死仓位，
  反而制造更大风险。

---

## 2. 风控引擎（`quantmind/risk`）

### 2.1 模块组成

| 模块 | 职责 |
|------|------|
| `risk/limits.py` | 限额定义（`RiskLimits`）、拒单代码（`RiskCode`）、判定结果（`RiskDecision`） |
| `risk/calendar.py` | 交易日历与交易时段（含期货夜盘、跨零点、节假日 2024–2026） |
| `risk/engine.py` | 风控引擎 + 熔断开关（`RiskEngine` / `RiskState`） |

### 2.2 限额档位

- **`default`** —— 偏保守的默认档（单笔 ≤100 手、单品种净持仓 ≤500、保证金率 ≤80%、
  日亏损率熔断 5%、回撤熔断 20%）。适用于百万级资金中低频组合，**上实盘前必须按账户规模重标定**。
- **`conservative`** —— 小资金 / 新策略首上实盘档（`RiskLimits.conservative()`）：单笔 ≤10、
  单品种 ≤50、保证金率 ≤30%、日亏损率 ≤2%、回撤 ≤10%、日下单 ≤200、分钟 ≤20。
- **`unlimited`** —— 仅单元测试 / 回放用，**禁止实盘**（所有检查置 `None`）。

### 2.3 两级熔断

- **`SOFT`**（默认自动触发形态）：触发日亏损 / 最大回撤 → **只允许减仓与平仓，禁止开仓**。
  出事先止血，但不强制在坏价位清仓。
- **`HARD`**：人工或严重异常触发（`halt(reason, "HARD")`）→ **所有委托一律拒绝**（含平仓），
  交由人工介入。
- 触发的判定：`RiskEngine.update_equity()` 在权益更新时检测日亏损 / 回撤，命中即 `_trigger()`。
- 解除：`resume(operator, note)` —— 必须显式调用并留痕；`reset_day()`（日切重置计数器）**不会**
  自动解除熔断。

### 2.4 拒单检查顺序（`check_order`）

`halt → symbol(黑白名单) → session(交易时段) → open_allowed → order_volume(手数/步进) →
price(偏离保护) → self_trade(防对敲) → close_volume(平仓超持仓) → position(净持仓上限) →
margin(保证金率) → rate(频率/日限)`。

返回第一个拒绝项；全过则返回 `RiskDecision.ok()`。通过单计入频率计数器，拒绝单计入审计日志。

---

## 3. 交易日历（`risk/calendar.py`）

- 内部时间统一 **UTC**，判断时换算北京时间（naive 视为 UTC）。
- **日盘**：商品期货 09:00–10:15/10:30–11:30/13:30–15:00；中金所股指 09:30–11:30/13:00–15:00、
  国债至 15:15；A股/港股/期权各有标准时段。
- **夜盘**：21:00 开盘，收盘分三档——黑色/化工/农产品 **23:00**、有色 **01:00**、贵金属+原油 **02:30**。
  **跨零点逻辑**：`crosses_midnight = night_end <= NIGHT_OPEN`；21:00 后需 `crosses_midnight or t<night_end`
  才判为交易；00:00~night_end 段归属前一日夜盘。0000–跨零点品种需前一日有夜盘（`has_night_session`
  要求次日也为交易日，节假日前最后一交易日晚不开夜盘）。
- **节假日**：内置 2024–2026 中国大陆休市日（参考值）；可用 `QM_HOLIDAY_FILE` 或
  `TradingCalendar.from_file()` 覆盖并**按交易所公告校准**。

---

## 4. 本地订单簿状态机（`live/order_manager.py`）

`OrderManager` 跟踪所有本地委托：

- 状态流转 `SUBMITTING → SUBMITTED → PARTTRADED → ALLTRADED`，终态 `CANCELLED/REJECTED`；
  **只向前流转**，乱序 / 重复回报去重。
- `active_orders()` / `active_requests()` 供风控自成交守卫与冻结量计算。
- `frozen_volume(vt_symbol)`：活动委托占用（用于资金/敞口估算）。
- `timeout_orders()` / `cancel_timeouts(gateway, now)`：挂单超时（默认 300s）自动撤单。
- `net_positions()`：由本地成交推算净持仓，供对账使用。

---

## 5. 对账（`live/reconcile.py`）

- `reconcile_positions(local, remote, tolerance)` / `reconcile_account(...)` 比对差异。
- `reconcile(local_positions, remote_positions, ..., risk_engine, halt_on_mismatch=True)`：
  本地推算持仓与网关查询持仓差异超容差 → **自动触发 SOFT 熔断**（禁开仓），防止错单累积。
- `LiveEngine.reconcile(remote_positions, remote_equity, halt_on_mismatch)` 是对外入口，
  应在定时心跳 / 收盘后周期调用。

---

## 6. 引擎接线

### 6.1 `LiveEngine`（`live/runner.py`）

- 构造：`LiveEngine(gateway, risk_engine=None, order_manager=None, initial_equity)`。
  `risk_engine=None` 自动创建**保守档**（`RiskLimits.conservative()`）；显式传 `False` **关闭风控**
  （仅内部测试，严禁实盘）。
- `send_order(req)`：**先 `risk.check_order()`**，拒单返回 `""`（不抛异常）；通过则网关发单 +
  `order_manager.add_order`。
- 回调：`on_bar/on_tick`（更新最新价）、`on_trade`（更新本地持仓 + `risk.on_trade`）、
  `on_account`（更新权益 + `risk.update_equity`，可能触发熔断）。
- 运维：`check_timeouts()`（撤超时单）、`reconcile()`（对账）、`status()`（Web/API 快照）。

### 6.2 `PaperEngine`（`paper/engine.py`）

- 同样前置风控检查（拒单记 `risk_rejected`、返回 `""`），`on_trade` 更新 `risk.on_trade`，
  `_mark_to_market` 调 `risk.update_equity`；`summary()` 含风控统计。模拟盘因此也受硬阈值约束，
  **不会裸奔**。

---

## 7. CLI 风控体检

```bash
python -m quantmind.cli risk --profile conservative --symbol rb2410 --exchange SHFE \
       --volume 5 --price 3500 --equity 1000000
```

输出：限额档、当前交易时段（实时北京时间/是否交易中/是否有夜盘）、以及一笔示例开仓与平仓委托的
拒/放行试算。若开仓被拒且代码为 `NOT_TRADING_TIME`，仅为当前非交易时段（时段闸门生效），并非配置错误。

---

## 8. 上线清单（Checklist）

- [ ] 用 `risk` 命令核对档位与时段，确认无预期外拒单。
- [ ] 按账户规模重标定 `RiskLimits`（手数、保证金率、日亏损 / 回撤线），**不要沿用 default**。
- [ ] 配置节假日覆盖文件（`QM_HOLIDAY_FILE`）并按交易所当期公告校准。
- [ ] 网关凭证走环境变量 / 密钥管理，不入库；先用 simnow/openctp 模拟盘跑通再接实盘。
- [ ] 部署 `check_timeouts()` 周期调用（行情心跳内）。
- [ ] 部署 `reconcile()` 周期调用（收盘后 / 定时），确认告警通道（Notifier）可用。
- [ ] 熔断解除 `resume()` 仅人工执行并留痕；审计日志（`RiskEngine.log`）落盘。
- [ ] `risk=False` 仅限回放测试，CI/实盘构建中不得出现。

---

## 9. 已知限制 / 待办

- 网关（`ctp_gateway` / `xtp_gateway` / `ib_gateway`）当前为桩；接真实券商需填凭证并跑通
  模拟盘。
- 日历节假日为参考值，需季度校准。
- 跨账户 / 跨网关组合级保证金核算尚未纳入（当前为单引擎视角）。
- 风控审计日志落盘与 Web 可视化面板（实时熔断状态）为后续打磨项。
