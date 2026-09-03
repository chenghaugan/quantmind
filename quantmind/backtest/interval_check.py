"""策略代码与数据周期兼容性校验。

检测策略代码中的时间逻辑与数据周期是否匹配，防止日内策略用日线数据回测。
"""
from __future__ import annotations

import ast
from typing import Dict, List


def check_strategy_interval_compatibility(code: str, interval: str) -> Dict[str, any]:
    """检查策略代码中的时间逻辑与数据周期是否匹配。
    
    Args:
        code: 策略源码
        interval: 数据周期（1d/1h/30m/15m/5m/1m）
    
    Returns:
        {compatible: bool, issues: list[str], suggestions: list[str]}
    """
    issues: List[str] = []
    suggestions: List[str] = []
    
    # 解析 AST 检测时间相关逻辑
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"compatible": True, "issues": [], "suggestions": []}
    
    # 检测时间相关属性访问
    time_patterns = {
        "hour": False,
        "minute": False,
        "second": False,
    }
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr in time_patterns:
                time_patterns[node.attr] = True
    
    has_intraday_logic = any(time_patterns.values())
    
    # 日线数据不支持日内逻辑
    if interval == "1d" and has_intraday_logic:
        issues.append(
            "策略代码包含日内时间判断（hour/minute），但数据周期为日线（1d）。"
            "日线数据的 datetime 通常是 00:00 或 15:00，不会触发 14:55 等分钟级判断。"
            "策略的日内平仓/止损逻辑将永远不会执行。"
        )
        suggestions.append("将数据周期改为内日数据：1h / 30m / 15m / 5m / 1m")
        suggestions.append("或移除日内时间判断逻辑，改为日线级别的出场规则（如持仓 N 天后平仓）")

    # 内日数据 + 策略引用 self.daily/self.mtf：正常（框架自动注入多周期上下文，无前视）
    uses_tf_ctx = "self.daily" in code or "self.mtf" in code
    if interval != "1d" and uses_tf_ctx:
        suggestions.append(
            "已启用日线级上下文（self.daily）：日线指标均基于当前交易日之前的"
            "已完成日线计算，无前视偏差。")

    # 日线数据 + 策略引用多周期上下文：日线周期下无意义
    if interval == "1d" and uses_tf_ctx:
        issues.append(
            "策略代码引用了 self.daily/self.mtf（多周期上下文），但数据周期本身为日线（1d）。"
            "多周期上下文仅在内日（分钟/小时）周期回测时由框架自动注入。")
        suggestions.append("移除 self.daily 相关查询，直接用日线数据计算指标；或改用内日数据周期。")
    
    # 内日数据但策略没有日内逻辑（可能是疏忽，仅警告）
    if interval != "1d" and not has_intraday_logic:
        # 检查是否有持仓天数等日线逻辑
        has_day_logic = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and "day" in node.id.lower():
                has_day_logic = True
                break
            if isinstance(node, ast.Attribute) and node.attr == "date":
                has_day_logic = True
                break
            # 使用 self.daily/self.mtf（多周期上下文注入）的策略本就是多周期混合结构
            if isinstance(node, ast.Attribute) and node.attr in ("daily", "mtf"):
                has_day_logic = True
                break
        
        if not has_day_logic:
            issues.append(
                f"数据周期为内日（{interval}），但策略代码未检测到时间相关逻辑。"
                "请确认策略是否充分利用了内日数据（如日内止盈止损、定时平仓）。"
            )
            suggestions.append("如果策略确实是日线级别，建议将数据周期改为 1d 以加快回测速度")
    
    compatible = len(issues) == 0
    return {
        "compatible": compatible,
        "issues": issues,
        "suggestions": suggestions,
    }
