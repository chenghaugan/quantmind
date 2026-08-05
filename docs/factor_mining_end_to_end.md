# 端到端因子挖掘流水线示例

本目录（`examples/`）提供一个**完整可运行**的示例脚本
`end_to_end_factor_pipeline.py`，把 QuantMind 的因子研究链路从头到尾跑一遍：

```
数据(合成面板) → LLM 挖掘(co/ea/tot) → 因子评测(IC/IR) → 去冗余(相关性聚类)
              → 权重优化(组合) → 复合 alpha 的样本外(OOS)多空回测 → 报告
```

同时验证了**真实 LLM 端到端驱动**（未配置 key 时自动降级离线 Mock）。

---

## 一、快速运行

在项目根目录：

```bash
# 离线 Mock（无 key、无网络也能跑通全链路）
.venv\Scripts\python.exe examples\end_to_end_factor_pipeline.py --seeds 1

# 指定搜索算法 / 更充分的合成面板
.venv\Scripts\python.exe examples\end_to_end_factor_pipeline.py --algo ea --n-dates 160 --n-symbols 12

# 强制用真实 LLM（需已配置 QM_LLM_API_KEY）
.venv\Scripts\python.exe examples\end_to_end_factor_pipeline.py --provider real --rounds 1
```

参数（`--help` 查看全量）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--n-dates` | 160 | 合成面板日期数 |
| `--n-symbols` | 12 | 合成面板标数 |
| `--seeds` | 3 | 每 seed 搜索迭代轮数 |
| `--algo` | `co` | 搜索算法 `co` \| `ea` \| `tot` |
| `--provider` | `auto` | `auto` \| `real` \| `mock` |
| `--rounds` | 取 `--seeds` | 覆盖搜索迭代轮数 |
| `--run-judge` | 关 | 用 LLM 对候选因子打分 |
| `--no-composite` | 开 | 关闭组合权重优化演示 |

---

## 二、脚本里的链路各步对应什么

| 步骤 | 脚本打印 | 对应模块 |
|---|---|---|
| 1 | `构建面板` | `factors/alpha_cs.Panel`（合成数据，确定性可复现） |
| 2 | `构造 provider` | `ai/provider.build_provider`（真实 LLM / Mock） |
| 3 | `端到端流水线` | `research.pipeline.run_pipeline` |
| 4 | `组合权重优化` | `research.combine.composite_backtest` |
| 5 | `报告` | 汇总：train/val/test IC、OOS Sharpe、权重 |

`run_pipeline` 内部完成**防泄漏切分**（train 用于搜索/去重/拟合，val 仅报告，
test 才是 OOS 回测），并逐代表因子做多空组合回测；`run_composite=True` 时再把
代表因子用 ICIR/最小方差等方案合成**复合 alpha** 并做 OOS 回测——这是
「因子研究」到「可交易组合」的最后一步。

---

## 三、怎么切到真实 LLM

`--provider auto`（默认）会自动读取已配置的 LLM（`SettingsService`，
来源优先级：`config/ai_settings.json` > 环境变量/`.env` 的 `QM_LLM_API_KEY`，见
`docs/real_data_setup.md` / 「设置」页）。有 key 用真实 LLM，无 key 回退 Mock。

- **真实 LLM** 会真正调用大模型产出多样化的因子表达式（跨算子嵌套、加减、除法等），
  而非 Mock 的关键词模板。
- 想让每次运行**可复现**、快速回归，就用 Mock：`--provider mock`。

> 建议先用 `--provider mock` 熟悉输出，再用 `--provider real --rounds 1` 做一次
> 真实端到端验证（rounds 越大越慢、越耗 token）。

---

## 四、关于合成数据与结果解读

脚本用**合成面板**（含趋势/周期/噪声，`numpy.random.default_rng(seed)` 确定性生成）
做演示，方便离线复现。要点：

1. **不要用这里的 Sharpe/IC 绝对值做投资决策**——合成数据里因子没有真实可交易的
   边际，OOS 结果多为噪声（正负随机）。
2. 这恰好体现**防泄漏设计的价值**：train 期高 IC 的因子，在 test 期（OOS）往往
   明显衰减甚至转负。看到 `train_ic` 高但 `OOS_sharpe` 差，是**链路在如实工作**，
   提醒你依赖样本外验证，而不是训练集 IC。
3. 换**真实行情**跑只需把合成面板换成 `Panel.from_bars(真实K线)`，其余链路不变。
   CLI 已提供真实数据入口：`quantmind factor-pipeline`（走 `DataManager` 取数）。

---

## 五、输出字段说明

- `train_ic / val_ic / test_ic`：各期截面 rank IC（Spearman）。
- `OOS_sharpe`：单个因子在 **test 期** 的多空组合年化 Sharpe（面板太小时可能为 `—`，
  采样不足无法分组）。
- `复合 alpha`：代表因子按最优权重合成的组合绩效（`sharpe / total_return /
  max_drawdown / fwd_IC`）。
- `权重`:各代表因子的组合权重（`icir` = 按 ICIR 加权，高分者权重大）。

---

## 六、真实行情秒级复用：本地行情仓库（Parquet 写缓存）

`/factor/pipeline` 若每次都用真实标的（如 `rb0/hc0/bu0/i0`），首次会走 AKShare
实时拉取（限频+回退，动辄数分钟）。为避免每次联网，`DataManager` 内置了
**本地行情仓库**（`data/store/disk_cache.py`，`DiskBarCache`）：

- **透明读缓存**：请求先查本地 `.parquet`（按 `symbol.exchange.interval` 分文件、
  存全量历史），命中即秒级返回（实测 `rb0.SHFE.1d` 4215 根 ≈ **0.17s**）。
- **自动回写**：只有**真实源**（非 mock）返回的结果才落盘；mock 合成数据**不入库**，
  避免污染仓库。
- **源判定**：`DataManager` 记录每个标的有实际由哪个源提供——
  命中磁盘缓存标 `disk_cache`，命中持久库标 `persistent_store`，否则标真实 feed / mock。
- **窗口切分**：仓库存全量历史，任意 `start/end` 窗口都直接从磁盘切片。

### 启用 / 配置
- 未显式配置时默认使用项目根 `data_cache/`（`app.py` lifespan 自动挂载），开箱即用。
- 也可在 `.env` 设 `QM_LOCAL_CACHE_ROOT=<目录>` 指定自定义位置。
- 强制忽略缓存重拉：构造 `DiskBarCache(root, refresh=True)`。

### 运营 / 清空
- 查看仓库概览：`GET /data/cache` → `{enabled, files, rows, last_datetime, root}`。
- 清空重建：`DELETE /data/cache`（下次请求自动从真实源重拉并重建）。
- 流水线响应里带 `cache` 字段（`files/rows/last_datetime`），前端 18 号页已展示。

### 典型耗时对比
| 场景 | 数据获取耗时 |
|---|---|
| 首次（真实源联网拉取，含限频回退） | 数分钟 |
| 二次及以后（命中本地仓库） | < 0.2 秒 |

> 注：流水线其余耗时（LLM 搜索 / 逐因子 OOS 回测）为**研究计算**，与数据取数无关；
> 数据取数已是秒级，缓存解决的是「每次重复联网拉行情」这一瓶颈。

---

## 七、相关代码路径

```
quantmind/
  research/
    pipeline.py              # run_pipeline 端到端流水线
    combine.py               # 组合构建 + 权重优化 + 复合回测
    dedup.py                 # 相关性聚类去冗余
    cross_sectional_backtest.py  # 因子→多空组合回测
    eval.py / evaluator.py   # 因子评测（IC/IR/分组）
    factors/panel_expr.py    # 面板 DSL 求值
  data/
    manager.py               # DataManager（查询链：本地仓库→持久库→数据源回退→回写）
    store/disk_cache.py      # 本地行情仓库（DiskBarCache，Parquet 写缓存）
  ai/provider.py            # LLM Provider（真实 / Mock）
examples/
  end_to_end_factor_pipeline.py   # 本示例
tests/
  test_combine.py            # 组合模块测试
  test_disk_cache.py         # 本地行情仓库测试
  test_pipeline_dedup.py     # 流水线测试
```
