# QuantMind 因子库

本文档汇总 QuantMind 内置因子库的族（family）结构、可用因子，以及借鉴 HKUDS/Vibe-Trading「5 大族」思路的扩展记录。

## 因子族总览

因子统一实现为 `quantmind.research.factors` 下的模块，每个因子是一个 `Factor` 子类（`compute(bars) -> pd.Series`，依赖 `wq.py` 时序原语，单标的日线 OHLCV 可稳定计算）。

| 族 | 模块 | 规模 | 说明 |
|---|---|---|---|
| technical | `technical.py` | 7 | 动量/均值回归/波动/量能/持仓/期限结构 |
| alpha101 | `alpha101.py` | 60 | WorldQuant Alpha101（单标的滚动近似） |
| alpha191 | `alpha191.py` | 10 | WorldQuant Alpha191 代表子集 |
| gtja191 | `gtja191.py` | 25 | 国泰君安短周期价量因子（A股风格） |
| qlib158 | `qlib158.py` | 20 | 常用量价技术指标（RSI/KDJ/MACD/布林带等） |
| academic | `academic.py` | 11 | 学术风格因子·价格代理版（FF/动量/BAB/特质波动等） |
| alpha_cs | `alpha_cs.py` | 50 | WorldQuant 截面（panel）因子 |
| futures_seat | `seat_futures.py` | 4 | 期货席位净持仓因子 |

**默认 `FactorRegistry` 注册** 94 个核心因子（technical 全 + alpha101 精选 22 + alpha191 精选 5 + gtja191 全 25 + qlib158 全 20 + academic 全 11 + futures_seat 4）。API（`/factor`、前端 FactorLibrary）经 `registry.get()` 解析，可直接取用注册因子。

## 借鉴 Vibe-Trading 的扩展（2026-08）

参照 HKUDS/Vibe-Trading 的「5 大族」（academic / alpha101 / fundamental / gtja191 / qlib158）因子库思想，本次为 QuantMind 新增 3 个因子族 + 补全 1 个既有族的核心子集：

### 1. gtja191（25 个，国泰君安短周期价量因子）
从 JoinQuant/GTJA 的 191 个短周期价量因子（存于工作区 `_ref_clone/alpha191`）中移植 25 个最经典、公式清晰、日线可稳定计算的代表（动量/反转/波动/量价相关/条件类）。公式用 `wq.py` 原语忠实重实现，RANK→`_rank`、CORR→`_corr`、REGBETA(SEQ)→`_slope`。
- 例：`gtja191_001`（量价背离取负）、`gtja191_014`（5日动量）、`gtja191_020`（6日收益）、`gtja191_060`（量价位置）、`gtja191_079`（RSI）、`gtja191_096`（KDJ RSV）。

### 2. qlib158（20 个，常用量价技术指标）
借鉴微软 qlib 常用指标，pandas 重实现：RSI(14)、KDJ(9,3,3) 三线、MACD(12,26,9) 三线、布林带(20,2) 三线、ATR(14)、CCI(20)、OBV、MOM(10)、ROC(12)、WR(14)、BIAS(10)、换手率代理、20日波动率、MA(20) 等。

### 3. academic（11 个，学术风格因子·价格代理版）
参照 Fama-French / Carhart / Frazzini-Pedersen / Ang 等文献，在单标的时序语境下实现**诚实的价格代理**（避免前视，仅用历史滚动窗口）：
- `acad_mom_12m_1m`（Carhart 动量）、`acad_short_term_reversal`（反转）、`acad_bab`（Betting-Against-Beta 代理）、`acad_idio_vol`（特质波动）、`acad_beta`、`acad_value_proxy`（价值代理）、`acad_liquidity_20`（Amihud 非流动性）、`acad_skew_20`（偏度）等。

### 4. alpha101 补全（40 → 60）
补齐 20 个缺失的 WorldQuant Alpha101 代表因子（alpha009/010/023/027/029/031/032/034/035/041/042/043/044/045/050/052/056/066/078/095）。

## 已知实现说明
- `wq.py::_reg_beta/_reg_resi` 曾受 pandas 3.0 `DataFrame.rolling().apply` 逐列化影响而异常（详见 commit `bf4879f`）。已改用「协方差/方差」等价实现（beta=Cov(y,x)/Var(x)，残差=y−(α+βx)），仅依赖逐列滚动原语，无前视；与 `academic._roll_beta` 数值偏差为 0。academic 的回归类实现与 gtja191 的 REGBETA（`_slope`）仍按各自语义保留。
- alpha101/gtja191 的 rank 采用单标的滚动时序近似（`_rank`），生产多标的面板场景建议替换为 `_rank_cs` 严格截面实现。

## 测试
- `tests/test_factor_families.py`：新族数量/冒烟/分类/registry 注册断言。
- `tests/test_wq_primitives.py`：`_reg_beta/_reg_resi` 回归保护（5 项断言，覆盖 pandas 3.0 多列 rolling 缺陷）。
- `tests/test_factors.py`、`tests/test_registry.py`、`tests/test_api.py`：既有回归。
