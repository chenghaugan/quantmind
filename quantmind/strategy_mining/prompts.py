"""策略挖掘 LLM 提示词。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


STRATEGY_ARCHITECT_SYSTEM = """你是一名量化策略架构师。你的任务是：

1. 分析用户提供的因子（含 IC/IR/Sharpe 等指标）
2. 选择最合适的策略模板
3. 设计入场/出场/仓位管理规则
4. 配置风险管理参数

你必须输出严格的 JSON 格式，符合以下 schema：

{
  "name": "策略名称（英文，如 'momentum_vol_target'）",
  "template": "策略模板类型（dual_ma | multifactor | vol_target | pair_trading）",
  "description": "策略描述（中文）",
  "factors": [
    {
      "name": "因子名称",
      "kind": "因子类型（momentum/mean_reversion/volatility 等）",
      "window": 窗口大小（整数）,
      "weight": 权重（浮点数，正负表示方向）,
      "icir": ICIR 值（从输入继承）
    }
  ],
  "params": {
    // 模板特定参数，例如：
    // dual_ma: {"fast": 5, "slow": 20}
    // multifactor: {"threshold": 0.3}
    // vol_target: {"lookback": 20, "target_vol": 0.20, "momentum_win": 60}
    // pair_trading: {"window": 30, "entry_z": 1.5, "exit_z": 0.3}
  },
  "risk": {
    "stop_loss": 止损比例（如 0.05 表示 5%）,
    "take_profit": 止盈比例（如 0.15 表示 15%）,
    "max_position": 最大仓位比例（0-1）
  },
  "symbol": "交易标的（如 'rb0'）",
  "exchange": "交易所（如 'SHFE'）",
  "capital": 初始资金（默认 1000000）,
  "rationale": "策略设计理由（中文，解释为什么选择这个模板和参数）"
}

选择模板的决策逻辑：
- 如果因子以动量/趋势为主 → dual_ma 或 vol_target
- 如果多个因子组合 → multifactor
- 如果涉及配对/价差 → pair_trading

参数设计原则：
- 基于因子的 IC/IR 调整权重（ICIR 越高，权重越大）
- 基于因子衰减调整窗口（半衰期长的因子用更大窗口）
- 基于波动率特征调整风控参数
"""


STRATEGY_ADJUSTMENT_SYSTEM = """你是一名量化策略优化专家。

基于回测失败结果，调整策略参数以通过绩效闸门。

调整原则：
1. Sharpe 过低 → 降低入场阈值（multifactor）或调整均线窗口（dual_ma）
2. 回撤过大 → 收紧止损、降低最大仓位
3. 胜率过低 → 提高入场阈值，减少噪音交易
4. 交易次数过少 → 降低阈值或缩短窗口

输出完整的 StrategySpec JSON，保持结构不变，只调整参数值。
"""


def build_architect_prompt(
    factors: List[Dict[str, Any]],
    constraint: Optional[str] = None,
    template_preference: Optional[str] = None,
) -> str:
    """构建策略架构师的用户提示。

    Args:
        factors: 因子列表（含指标）
        constraint: 用户约束（如"偏动量"、"低换手"）
        template_preference: 模板偏好

    Returns:
        格式化的用户提示字符串
    """
    parts: List[str] = []

    # 因子信息
    parts.append("## 可用因子\n")
    for i, f in enumerate(factors, 1):
        parts.append(
            f"{i}. **{f.get('name', 'unknown')}**\n"
            f"   - 类型：{f.get('kind', 'unknown')}\n"
            f"   - 窗口：{f.get('window', 'N/A')}\n"
            f"   - IC 均值：{f.get('ic_mean', 'N/A')}\n"
            f"   - ICIR: {f.get('icir', 'N/A')}\n"
            f"   - Sharpe: {f.get('sharpe', 'N/A')}\n"
            f"   - 表达式：{f.get('expression', '内置')}\n"
        )

    # 用户约束
    if constraint:
        parts.append(f"\n## 用户约束\n{constraint}\n")

    # 模板偏好
    if template_preference:
        parts.append(f"\n## 模板偏好\n优先使用 `{template_preference}` 模板\n")

    # 任务说明
    parts.append(
        "\n## 任务\n"
        "请基于以上因子设计一个可执行的交易策略。分析因子的特性（IC/IR/衰减/单调性），"
        "选择最合适的策略模板，设计参数，并输出符合 schema 的 JSON。\n"
    )

    return "\n".join(parts)


def build_adjustment_prompt(
    spec_dict: Dict[str, Any],
    sharpe: float,
    max_drawdown: float,
    annual_return: float,
    win_rate: float,
    min_sharpe: float,
    max_drawdown_threshold: float,
    iteration: int,
) -> str:
    """构建参数调整的用户提示。

    Args:
        spec_dict: 当前 StrategySpec 字典
        sharpe: 当前 Sharpe
        max_drawdown: 当前最大回撤
        annual_return: 年化收益
        win_rate: 胜率
        min_sharpe: 目标最低 Sharpe
        max_drawdown_threshold: 目标最大回撤下限
        iteration: 当前迭代次数

    Returns:
        格式化的调整提示字符串
    """
    return f"""## 当前策略规格
{spec_dict}

## 回测结果
- Sharpe Ratio: {sharpe:.4f} (目标：≥ {min_sharpe})
- 最大回撤：{max_drawdown:.4f} (目标：≥ {max_drawdown_threshold})
- 年化收益：{annual_return:.4f}
- 胜率：{win_rate:.4f}

## 任务
这是第 {iteration} 次迭代。策略未通过闸门。请分析失败原因，调整参数，
输出新的 StrategySpec JSON。

调整原则：
- 如果 Sharpe 过低，考虑降低阈值或调整因子权重
- 如果回撤过大，考虑收紧止损或降低仓位
- 保持策略逻辑不变，只调整参数

输出完整的 StrategySpec JSON。
"""
