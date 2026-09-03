"""BacktestService: 回测 / WalkForward / 策略清单"""
import asyncio
import inspect
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd

from ...core.constant import Exchange, Interval
from ...core.contracts import default_size
from ...core.engine import EventEngine
from ...core.event import Event, EventType
from ...data.feed.base import HistoryRequest
from ...data import DataManager
from ...strategy import (
    run_strategy,
    MultiFactorStrategy,
    DualMaStrategy,
    VolTargetStrategy,
    PairTradingStrategy,
    CtaTemplate,
)
from ...strategy.components import ComposableStrategy
from ...strategy.mined import ChanThirdBuyStrategy
from ...strategy.validation import (
    MomentumCtaStrategy,
    ChanFirstBuyStrategy,
    BollingerRecoverStrategy,
    DEFAULT_SETTINGS as _VALIDATION_DEFAULTS,
)
from ...backtest.walkforward import walk_forward
from ...research.risk_xray import compute_risk_xray
from ..schemas import BacktestRequest, WalkForwardRequest, StrategyInfo
from ..ws import manager as ws_manager


_logger = logging.getLogger("quantmind.api")


_STRATEGY_MAP = {
    "dual_ma": DualMaStrategy,
    "multifactor": MultiFactorStrategy,
    "vol_target": VolTargetStrategy,
    "pair": PairTradingStrategy,
    "composable": ComposableStrategy,
    # 策略验证（单品种 idea → 回测 → 门槛）确定性模板
    "momentum": MomentumCtaStrategy,
    "chan_first_buy": ChanFirstBuyStrategy,
    "chan_third_buy": ChanThirdBuyStrategy,
    "bollinger_recover": BollingerRecoverStrategy,
}


def _daily_close_times(daily_bars: List[Any]) -> List[datetime]:
    """日线 bar 的可见时间 = 交易日次日凌晨 00:00 UTC（防前视）。

    注意：BarData.datetime 是标准库 datetime（没有 pandas 的 normalize 方法），
    归零须用 .replace()。历史数据可能混有 16:00 UTC 旧约定，统一归到当日 00:00。
    """
    return [
        b.datetime.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        for b in daily_bars
    ]


def _safe_num(x) -> "float | None":
    """安全取数值（NaN/None/非法 → None），用于门槛判定指标提取。"""
    if x is None:
        return None
    try:
        f = float(x)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _sanitize(o: Any) -> Any:
    """递归把非有限 float 转为 None，避免 JSON 序列化抛错"""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(x) for x in o]
    return o


def _strategy_codegen_system() -> str:
    """LLM 策略编程系统提示词（正式生成与对话草稿共用）。"""
    return (
            "你是资深量化策略程序员。请把用户的「策略思想」实现为一个继承 "
            "CtaTemplate 的确定性策略类。\n"
            "硬性约束：\n"
            "1. 类名以 Strategy 结尾（如 IdeaStrategy），且必须继承 CtaTemplate。\n"
            "2. 只允许 import：from quantmind.strategy.base import CtaTemplate；\n"
            "   from quantmind.core.utility import ArrayManager；numpy/pandas/math。\n"
            "3. 用 self.am = ArrayManager(size)；self.am.update_bar(bar)；\n"
            "   self.am.close / self.am.high / self.am.low 为 list。\n"
            "   bar 对象属性为 bar.open_price / bar.high_price / bar.low_price /\n"
            "   bar.close_price（注意：没有 bar.high / bar.close 这种短名）。\n"
            "   分钟级回测时框架自动注入 self.mtf（多周期上下文）与 self.daily（=1d），\n"
            "   均无前视（只用已完成 bar；可用周期清单见【数据周期】说明）：\n"
            "   self.mtf.tf('1h', bar.datetime).close[-1]  截至当前的上一根1h收盘\n"
            "   self.mtf.tf('1h', bar.datetime).sma(20)    1h均线（数据不足返回 None，请判空）\n"
            "   self.mtf.tf('1d', bar.datetime).prev_high()  前一交易日高点\n"
            "   self.daily.prev_high(bar.datetime) / prev_close / sma(20, ...) 为 1d 简化写法\n"
            "   若策略思想包含更高周期规则（日线定方向、前日高低点等），优先用上述上下文实现。\n"
            "4. 用 self.set_target(bar.vt_symbol, target) 下单（target=正多/负空/0空仓）。\n"
            "5. 参数写在 parameters 列表，并在 __init__ 给默认值，\n"
            "   形如 self.window = 20（之后 super().__init__(context, setting)）。\n"
            "6. 禁止：exec/eval/open/网络/文件读写/线程；禁止 import 其他模块。\n"
            "   禁止使用 getattr/setattr/hasattr（沙箱白名单不允许）：\n"
            "   self.daily/self.mtf 由框架保证已注入（分钟级回测），直接访问即可，\n"
            "   不要写 getattr(self, 'daily', None) 之类的防御性兜底。\n"
            "7. 逻辑必须忠实于用户描述，不要自行发明多余规则。\n"
            "   框架**没有内置**止盈/止损/定时平仓：用户思想里的每一条规则\n"
            "   （指标计算、入场条件、止盈止损、时间出场）都必须在 on_bar 里\n"
            "   用代码显式实现，不得只写参数而不写逻辑。\n"
            "8. 请在代码末尾额外输出参数搜索范围（供参数优化用）：\n"
            "   PARAM_GRID = {\"window\": [10, 20, 30], \"stop_loss\": [0.03, 0.05]}\n"
            "   每个参数 2~4 个候选值，围绕你的默认参数展开；只含数值参数。\n"
            "只输出代码本身，不要任何解释或 markdown 围栏。\n\n"
            "常用实现模式：\n"
            "- 百分比止盈止损：记录入场价，触发后平仓。例如持多时\n"
            "    if self.entry_price and closes[-1] <= self.entry_price * (1 - self.stop_loss):\n"
            "        self.set_target(bar.vt_symbol, 0)\n"
            "- 日内平仓（不持隔夜仓）：**严禁硬编码具体分钟**（如 14:55/06:55）——\n"
            "    K线粒度不同（1m/5m/15m/30m/1h），每天最后一根K线的时刻各不相同，\n"
            "    硬编码时刻在较粗粒度下可能永不触发，导致持仓跨越数月。\n"
            "    稳健做法：在 __init__ 里初始化 self._last_date = None，\n"
            "    on_bar 开头检测日期变化，新交易日第一根K线先平掉昨日残留仓位：\n"
            "    cur_date = bar.datetime.date()\n"
            "    if self._last_date is not None and cur_date != self._last_date:\n"
            "        self.set_target(bar.vt_symbol, 0)  # 平掉昨日残留仓位\n"
            "    self._last_date = cur_date\n"
            "- 布林带（窗口 N、K 倍标准差）：mean = sum(closes[-N:]) / N；\n"
            "    std = statistics 或 numpy 计算；上轨 = mean + K*std；下轨 = mean - K*std。\n\n"
            "示例结构：\n"
            "from quantmind.strategy.base import CtaTemplate\n"
            "from quantmind.core.utility import ArrayManager\n\n"
            "class IdeaStrategy(CtaTemplate):\n"
            "    parameters = ['window', 'size', 'max_pos']\n\n"
            "    def __init__(self, context, setting=None):\n"
            "        self.window = 20\n"
            "        self.size = 1\n"
            "        self.max_pos = 1.0\n"
            "        self.am = None\n"
            "        self.last_target = 0.0\n"
            "        super().__init__(context, setting)\n\n"
            "    def on_bar(self, bar):\n"
            "        if self.am is None:\n"
            "            self.am = ArrayManager(self.window + 5)\n"
            "        self.am.update_bar(bar)\n"
            "        if not self.am.inited:\n"
            "            return\n"
            "        closes = self.am.close\n"
            "        # ... 你的规则，用 self.set_target(bar.vt_symbol, target)\n"
)


def _strip_code_fences(source: str) -> str:
    """去除源码外围的 Markdown 代码围栏（```python ... ```），使落库/持久化的代码可编译。"""
    s = (source or "").strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines)
    return s.strip()


def _strategy_class_name(source: str) -> str:
    """从策略源码提取主类名（优先取以 Strategy 结尾的类）。"""
    import re as _re

    names = _re.findall(r"class\s+(\w+)", source or "")
    if not names:
        return ""
    for n in names:
        if n.endswith("Strategy"):
            return n
    return names[-1]



_DRAFT_REPAIR_ROUNDS = 2  # 沙箱自修复轮数上限（+首次生成，最多3次LLM调用）


class BacktestService:
    def __init__(self, dm: DataManager, ee: EventEngine):
        self.dm = dm
        self.ee = ee
        # 动态注册的 AI 生成策略（仅实例级，避免全局污染）
        self._extra_strategies: Dict[str, type] = {}
        # 耐重启：启动时把 knowledge.db 已沉淀/已注册策略载入运行池
        self._load_persisted_strategies()

    def register_generated_strategy(self, name: str, source: str) -> Tuple[bool, str, dict]:
        """注册 AI 生成策略源码 -> 实例级策略池。

        先经沙箱二次校验（compile_strategy），通过后才在隔离命名空间 exec，
        从其中找 CtaTemplate 子类存入 ``self._extra_strategies``。

        :return: (ok, err, info)；info = {"name", "parameters"}
        """
        from ...ai.sandbox import compile_strategy

        source = _strip_code_fences(source)

        ok, err, _ = compile_strategy(source)
        if not ok:
            return False, err or "代码未通过沙箱校验", {}
        try:
            from ...ai.sandbox import restricted_globals

            ns: Dict = restricted_globals()
            exec(compile(source, "<generated>", "exec"), ns, ns)
        except Exception as exc:  # noqa: BLE001
            return False, f"执行策略源码失败: {exc}", {}
        cls = None
        for v in ns.values():
            if inspect.isclass(v) and issubclass(v, CtaTemplate) and v is not CtaTemplate:
                cls = v
                break
        if cls is None:
            # 兜底：放宽到名字以 Strategy 结尾的类
            for v in ns.values():
                if inspect.isclass(v) and (v.__name__.endswith("Strategy") or "Strategy" in v.__name__):
                    cls = v
                    break
        if cls is None:
            return False, "未找到策略类", {}
        self._extra_strategies[name] = cls
        # 提取 LLM 同步输出的参数搜索范围（参数优化用）：仅保留已声明参数中的数值候选
        raw_grid = ns.get("PARAM_GRID")
        param_grid: Dict[str, List[float]] = {}
        if isinstance(raw_grid, dict):
            allowed = set(getattr(cls, "parameters", []) or [])
            for k, vals in raw_grid.items():
                if (k in allowed and isinstance(vals, (list, tuple)) and vals
                        and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                for v in vals)):
                    param_grid[k] = list(vals)[:6]
        return True, "", {"name": name, "parameters": list(getattr(cls, "parameters", [])),
                          "param_grid": param_grid}

    def _load_persisted_strategies(self) -> None:
        """启动/重建时把已沉淀策略载入运行池（耐重启）。

        来源分两类：
        1. 规范适配模块 ``quantmind.strategy.mined`` 的 ``MINED_STRATEGIES``（历史挖掘策略的可运行重写）；
        2. knowledge.db 中已注册/挖掘的策略（``lifecycle`` 表 + ``strategies`` 表）。
        历史代码可能因模块路径漂移编译失败 -> 跳过，不影响其它策略加载。
        """
        # 1) 规范适配策略：稳定可运行，始终注册（按类名）
        try:
            from ...strategy.mined import MINED_STRATEGIES as _mined

            seen = set(self._extra_strategies)
            for name, cls in _mined.items():
                if name in seen:
                    continue
                self._extra_strategies[name] = cls
                seen.add(name)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("加载规范适配策略失败(忽略): %s", exc)

        if not self._extra_strategies:
            return
        seen = set(self._extra_strategies)
        try:
            from ...knowledge.store import KnowledgeStore

            ks = KnowledgeStore()
            # 2) 已注册生命周期策略（有明确名称）
            for lc in ks.list_strategy_lifecycles(limit=300):
                name = (lc.get("strategy_id") or "").strip()
                code = lc.get("code") or ""
                if not name or not code or name in seen:
                    continue
                ok, _, _ = self.register_generated_strategy(name, code)
                if ok:
                    seen.add(name)
            # 3) 研究挖掘脚本（从类名取名）
            for rec in ks.list_mined_strategies(limit=300):
                name = _strategy_class_name(rec.get("code") or "")
                code = rec.get("code") or ""
                if not name or name in seen or not code:
                    continue
                ok, _, _ = self.register_generated_strategy(name, code)
                if ok:
                    seen.add(name)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("加载已沉淀策略失败(忽略): %s", exc)

    def _resolve_strategy_class(self, name: str):
        """按名解析策略类：内置映射 -> 运行池 -> 知识库惰性加载 -> 规范适配模块。"""
        cls = _STRATEGY_MAP.get(name) or self._extra_strategies.get(name)
        if cls is not None:
            return cls
        # 惰性：从库中按名载入（例如 list 已展示但尚未进池的挖掘/注册策略）
        try:
            from ...knowledge.store import KnowledgeStore

            ks = KnowledgeStore()
            for lc in ks.list_strategy_lifecycles(limit=300):
                if (lc.get("strategy_id") or "").strip() == name:
                    ok, _, _ = self.register_generated_strategy(name, lc.get("code") or "")
                    return self._extra_strategies.get(name)
            for rec in ks.list_mined_strategies(limit=300):
                if _strategy_class_name(rec.get("code") or "") == name:
                    ok, _, _ = self.register_generated_strategy(name, rec.get("code") or "")
                    return self._extra_strategies.get(name)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("按名加载策略失败(忽略): %s", exc)
        # 兜底：规范适配模块
        try:
            from ...strategy.mined import MINED_STRATEGIES as _mined

            cls = _mined.get(name)
            if cls is not None:
                self._extra_strategies[name] = cls
                return cls
        except Exception:  # noqa: BLE001
            pass
        return None

    def list_strategies(self) -> List[StrategyInfo]:
        result = [
            StrategyInfo(
                name="dual_ma",
                description="双均线趋势/动量策略",
                parameters={"fast": 5, "slow": 20, "size": 1, "max_pos": 1.0},
            ),
            StrategyInfo(
                name="multifactor",
                description="多因子组合策略（动量+均值回复+波动率）",
                parameters={"specs": "see research", "threshold": 0.3, "size": 1, "max_pos": 1.0},
            ),
            StrategyInfo(
                name="vol_target",
                description="全天候风格：波动率目标+动量过滤(单标的风险平价)",
                parameters={"lookback": 20, "target_vol": 0.20, "momentum_win": 60, "size": 1, "max_pos": 1.0},
            ),
            StrategyInfo(
                name="pair",
                description="配对交易：价差合成标的 z-score 均值回复",
                parameters={"window": 30, "entry_z": 1.5, "exit_z": 0.3, "size": 1, "max_pos": 1.0},
            ),
            StrategyInfo(
                name="composable",
                description="5 组件可组合策略（Alpha/Portfolio/Risk/Execution 可插拔）",
                parameters={"alpha": "MultiFactorAlpha/MomentumAlpha", "risk": "NullRisk/RiskGateModel"},
            ),
        ]
        for k in self._extra_strategies:
            result.append(StrategyInfo(name=k, description="AI 生成策略", parameters={"size": 1}))
        # 追加研究挖掘的策略（即使历史代码当前不可编译，也展示供查看/重建）
        seen = {s.name for s in result}
        try:
            from ...knowledge.store import KnowledgeStore

            ks = KnowledgeStore()
            for rec in ks.list_mined_strategies(limit=100):
                name = _strategy_class_name(rec.get("code") or "")
                if not name or name in seen:
                    continue
                seen.add(name)
                result.append(StrategyInfo(
                    name=name,
                    description=("AI挖掘:" + str(rec.get("idea") or "")[:40])
                    if rec.get("idea") else "AI 挖掘策略",
                    parameters={"size": 1, "source": "mined"},
                ))
        except Exception:  # noqa: BLE001
            pass
        return result

    async def run_backtest(self, req: BacktestRequest) -> Dict[str, Any]:
        strat_class = self._resolve_strategy_class(req.strategy)
        if strat_class is None:
            # 未知策略名必须报错，不得静默回退（否则测 A 策略返回 B 策略的结果）
            return {"error": f"未知策略: {req.strategy}"}
        vt = f"{req.symbol}.{req.exchange.upper()}"

        # 发送开始事件
        await ws_manager.broadcast({
            "type": "backtest_start",
            "strategy": req.strategy,
            "symbol": req.symbol,
            "exchange": req.exchange,
            "mode": req.mode,
            "timestamp": datetime.now().isoformat(),
        })
        
        try:
            # 构造 HistoryRequest，支持可选日期范围
            history_kwargs = {
                "symbol": req.symbol,
                "exchange": Exchange(req.exchange.upper()),
                "interval": Interval("1d"),
            }
            if req.start:
                from datetime import datetime as _dt
                history_kwargs["start"] = _dt.fromisoformat(req.start)
            if req.end:
                from datetime import datetime as _dt
                history_kwargs["end"] = _dt.fromisoformat(req.end)
            bars = await self.dm.get_bar_data(HistoryRequest(**history_kwargs))
            if not bars:
                _logger.warning(f"回测无数据: {req.symbol}.{req.exchange}")
                await ws_manager.broadcast({
                    "type": "backtest_error",
                    "strategy": req.strategy,
                    "error": "无数据",
                })
                return {"error": "无数据"}
            
            # 发送进度事件（数据加载完成）
            await ws_manager.broadcast({
                "type": "backtest_progress",
                "strategy": req.strategy,
                "progress": 0.3,
                "message": f"已加载 {len(bars)} 根K线",
            })
            
            sizes = dict(req.sizes) or {vt: default_size(vt)}
            result = await asyncio.to_thread(
                run_strategy,
                req.mode,
                strat_class,
                vt,
                dict(req.setting),
                bars,
                self.ee,
                sizes,
                req.gateway,
                None,
                req.cost,
            )
            _logger.info(f"回测完成: {req.strategy} on {vt}, {len(bars)} bars")
            
            # 计算风险 X 光指标
            result_sanitized = _sanitize(result)
            try:
                equity_curve = _extract_equity_curve(result)
                # trades/positions 可能为计数而非明细，归一化为 compute_risk_xray 所需结构
                _raw_trades = result.get("trades", [])
                trades = _raw_trades if isinstance(_raw_trades, list) else []
                _raw_pos = result.get("positions", {})
                positions = _raw_pos if isinstance(_raw_pos, dict) else {}
                if equity_curve is not None and len(equity_curve) > 0:
                    risk_xray = compute_risk_xray(equity_curve, trades, positions)
                    result_sanitized["risk_xray"] = risk_xray.to_dict()
                    _logger.info(f"风险 X 光已生成: 夏普={risk_xray.sharpe_ratio:.2f}, 最大回撤={risk_xray.max_drawdown:.2%}")
            except Exception as exc:  # noqa: BLE001
                _logger.warning(f"风险 X 光计算失败: {exc}")
            
            # 发送完成事件
            await ws_manager.broadcast({
                "type": "backtest_complete",
                "strategy": req.strategy,
                "progress": 1.0,
                "message": "回测完成",
                "summary": result_sanitized.get("summary", {}),
            })
            
            return result_sanitized
        except Exception as e:
            _logger.error(f"回测失败: {req.strategy} on {vt} - {str(e)}", exc_info=True)
            await ws_manager.broadcast({
                "type": "backtest_error",
                "strategy": req.strategy,
                "error": str(e),
            })
            return {"error": f"回测失败: {str(e)}"}


    async def validate_strategy(self, req, provider=None, progress=None) -> Dict[str, Any]:
        """策略思想测试：策略思路 →（LLM 预编程 或 预置模板）→ 多品种真实回测 → 门槛 → 有效策略库。

        流程：
          1. 策略来源：``req.code`` 非空时直接使用用户审定的代码（AST 沙箱校验后注册）；
             否则把 ``req.idea``（策略思想）交给 LLM 预编程为 CtaTemplate 策略代码。
          2. ``req.symbols`` 多品种：逐品种独立拉数据、回测（自动带合约乘数）。
          3. ``req.gate`` 非空时对**每个品种**用 judge_strategy 规则判定；
             达标品种（verified）且 ``req.promote`` → 自动写入 lifecycle。
        失败闭合：任何异常只返回带 ``error`` 的 dict，不抛异常。
        """
        from ...strategy.validation import DEFAULT_SETTINGS
        from ...research.knowledge_loop import judge_strategy

        # ------------------------------------------------------- 0) 参数优化分支（三防线）
        _opt = getattr(req, "optimization", None)
        _opt_enabled = bool(_opt) and (
            _opt.get("enabled", False) if isinstance(_opt, dict)
            else bool(getattr(_opt, "enabled", False)))
        if _opt_enabled:
            return await self._validate_with_optimization(req, provider=provider,
                                                          progress=progress)

        def _prog(msg: str, cur: int = 0, tot: int = 0) -> None:
            if progress:
                progress(msg, cur, tot)

        # ------------------------------------------------------- 1) 标的与周期
        symbols = [x for x in (getattr(req, "symbols", None) or []) if x and x.strip()]
        if not symbols and getattr(req, "symbol", ""):
            symbols = [getattr(req, "symbol", "")]
        if not symbols:
            return {"error": "未指定标的（symbols 为空）"}

        # 多周期：req.intervals 非空时逐周期回测（兼容旧单 interval 字段）
        _valid_ivs = ("1d", "1h", "30m", "15m", "5m", "1m")
        intervals = [iv for iv in (getattr(req, "intervals", None) or [])
                     if iv in _valid_ivs]
        if not intervals:
            intervals = [req.interval if req.interval in _valid_ivs else "1d"]

        # ------------------------------------------------------- 2) 策略来源
        generated_code: str = ""
        _prog("LLM 编程中（把策略思想翻译为代码）…", 0, 1)
        approved = (getattr(req, "code", "") or "").strip()
        if approved:
            # 用户在对话式编程阶段审定的代码：跳过 LLM，直接注册（仍过沙箱校验）
            _prog("使用审定代码…", 0, 1)
            code = approved
        else:
            code, err = await self._llm_generate_strategy(provider, req.idea,
                                                          interval=intervals[0])
            if err:
                return {"error": f"LLM 策略编程失败：{err}"}
        # 生成时左移校验：纯日线周期下多周期上下文无数据来源，直接报错（失败左移）
        if intervals == ["1d"] and ("self.mtf" in code or "self.daily" in code):
            return {"error": ("策略代码引用了 self.mtf/self.daily（多周期上下文），"
                              "但本次数据周期为 1d，二者不兼容。请修改策略思想或更换数据周期。")}
        name = "idea_strategy"
        ok, err2, _info = self.register_generated_strategy(name, code)
        if not ok:
            return {"error": f"策略注册失败：{err2}"}
        cls = self._extra_strategies.get(name)
        generated_code = code
        strategy_desc = "用户审定的 LLM 策略" if approved else "LLM 预编程策略"

        _prog("数据加载与回测准备…", 0, len(symbols))
        setting = dict(req.setting or {}) or dict(DEFAULT_SETTINGS.get(name) or {})

        per_symbol: list = []
        _total = len(symbols) * len(intervals)
        _idx = 0
        for iv in intervals:
            for _i, sym in enumerate(symbols, 1):
                _idx += 1
                _prog(f"回测 {sym}@{iv}（{_idx}/{_total}）…", _idx - 1, _total)
                vt = f"{sym}.{req.exchange.upper()}"
                try:
                    bars = await self.dm.get_bar_data(HistoryRequest(
                        symbol=sym,
                        exchange=Exchange(req.exchange.upper()),
                        interval=Interval(iv),
                        start=datetime.fromisoformat(req.start) if req.start else None,
                        end=datetime.fromisoformat(req.end) if req.end else None,
                    ))
                except Exception as exc:  # noqa: BLE001
                    per_symbol.append({"symbol": sym, "interval": iv, "error": f"数据获取失败：{exc}"})
                    continue
                if not bars:
                    per_symbol.append({"symbol": sym, "interval": iv, "error": "无数据（检查 data_cache）"})
                    continue
                if len(bars) < 200:
                    per_symbol.append({"symbol": sym, "interval": iv, "error": f"数据不足（{len(bars)} 根 < 200）"})
                    continue

                # 多周期上下文：近周期从回测 bars 重采样（同源对齐），
                # 日线/周线独立拉取（深度不受基础周期限制），全部无前视。
                daily_ctx = None
                mtf = None
                _needs_tf = generated_code and (
                    "self.daily" in generated_code or "self.mtf" in generated_code)
                if iv != "1d":
                    try:
                        _daily_bars = await self.dm.get_bar_data(HistoryRequest(
                            symbol=sym,
                            exchange=Exchange(req.exchange.upper()),
                            interval=Interval.DAILY,
                            start=None,
                            end=datetime.fromisoformat(req.end) if req.end else None,
                        ))
                        if _daily_bars:
                            from ...strategy.daily_context import DailyContext
                            from ...strategy.multi_tf import MultiTFContext, resample_bars
                            daily_ctx = DailyContext(_daily_bars)
                            mtf = MultiTFContext()
                            mtf.add("1d", _daily_bars,
                                    close_times=_daily_close_times(_daily_bars))
                            _wb, _wb_ct = resample_bars(_daily_bars, "1w", "1d")
                            if _wb:
                                mtf.add("1w", _wb, close_times=_wb_ct)
                    except Exception:  # noqa: BLE001 —— 日线上下文缺失走下方显式错误
                        _logger.exception("日线上下文构建失败 iv=%s", iv)
                        daily_ctx = None
                # 依赖多周期上下文的代码在日线数据不可用 → 显式错误（绝不静默崩溃）
                if _needs_tf and daily_ctx is None:
                    per_symbol.append({
                        "symbol": sym, "interval": iv,
                        "error": "策略依赖日线级上下文（self.daily/self.mtf），"
                                 f"但品种 {sym} 的日线数据不可用",
                    })
                    continue
                # 近周期重采样（必须在线日数据就绪后，基于回测自身 bars 构建）
                if mtf is not None:
                    from ...strategy.multi_tf import RESAMPLE_CANDIDATES, resample_bars
                    for tgt in RESAMPLE_CANDIDATES.get(iv, []):
                        try:
                            _rb, _rct = resample_bars(bars, tgt, iv)
                            mtf.add(tgt, _rb, close_times=_rct)
                        except Exception:  # noqa: BLE001
                            pass

                # 周期-策略兼容性校验（防止日内策略用日线数据）
                if generated_code:
                    from ...backtest.interval_check import check_strategy_interval_compatibility
                    compat = check_strategy_interval_compatibility(generated_code, iv)
                    if not compat["compatible"]:
                        per_symbol.append({
                            "symbol": sym,
                            "error": "策略与数据周期不兼容：" + "；".join(compat["issues"]),
                            "suggestions": compat["suggestions"],
                        })
                        continue

                sizes = {vt: default_size(vt)}
                try:
                    result = await asyncio.to_thread(
                        run_strategy, "backtest", cls, vt, setting, bars,
                        self.ee, sizes, "ctp", None, req.cost,
                        daily_context=daily_ctx, mtf_context=mtf,
                    )
                except Exception as exc:  # noqa: BLE001
                    per_symbol.append({"symbol": sym, "interval": iv, "error": f"回测失败：{exc}"})
                    continue
                report = _sanitize(result.get("report") or {})

                # --------------------------------------------------- 3) 逐品种门槛
                item: Dict[str, Any] = {
                    "symbol": sym,
                    "interval": iv,
                    "exchange": req.exchange.upper(),
                    "bars": len(bars),
                    "report": report,
                    "equity_curve": _sanitize(result.get("equity_curve") or []),
                    "trades": result.get("trades", 0),
                    "trade_list": _sanitize(result.get("trade_list") or []),
                    "benchmark_curve": _sanitize(result.get("benchmark_curve") or []),
                }
                if req.gate:
                    gate = dict(req.gate)
                    sharpe = _safe_num(report.get("sharpe"))
                    mdd = _safe_num(report.get("max_drawdown"))
                    total_ret = _safe_num(report.get("total_return"))
                    total_cost = _safe_num(report.get("total_cost"))
                    cost_ratio = _safe_num(report.get("cost_ratio"))
                    judge = await judge_strategy(
                        None, {"run_id": vt, "state": "BACKTEST", "status": "",
                               "sharpe": sharpe, "max_drawdown": mdd},
                        gate=gate, fallback_rules=True)
                    status = judge.get("status")
                    reason = judge.get("reason", "")
                    # 高换手成本拦截：启用成本且成本/净收益超阈值 → 拒绝入库（即使零成本 Sharpe 高）
                    max_cost_ratio = gate.get("max_cost_ratio", 0.6) or 0.6
                    if req.cost and max_cost_ratio > 0 and cost_ratio > max_cost_ratio:
                        status = "rejected"
                        reason = (reason + "；" if reason else "") + \
                            f"成本/净收益 {cost_ratio:.1%} 超上限 {max_cost_ratio:.0%}（高换手，总成本 {total_cost:.0f}）"
                    item["gate"] = {
                        "enabled": True,
                        "status": status,
                        "reason": reason[:300],
                        "tags": judge.get("tags") or [],
                        "metrics": {"sharpe": sharpe, "max_drawdown": mdd,
                                    "total_return": total_ret,
                                    "total_cost": total_cost, "cost_ratio": cost_ratio},
                    }
                per_symbol.append(item)

        out: Dict[str, Any] = {
            "idea": req.idea or "",
            "strategy": name,
            "strategy_desc": strategy_desc,
            "code": generated_code,
            "interval": intervals[0],
            "intervals": intervals,
            "per_symbol": per_symbol,
            "gate_enabled": bool(req.gate),
        }

        # --------------------------------------------------- 4) 达标品种入库
        if req.gate and req.promote:
            self._promote_verified(req, per_symbol, generated_code, out)
        return out

    def delete_validation_history(self, run_id: str) -> Dict[str, Any]:
        """删除一条策略验证历史记录，并尝试删除对应的 lifecycle 入库记录。

        返回 {deleted_history: bool, deleted_lifecycle: bool, strategy_id: str|None}
        """
        from pathlib import Path
        history_file = (Path(__file__).resolve().parent.parent.parent.parent
                        / "data_cache" / "strategy_validation_runs.json")
        deleted_history = False
        deleted_lifecycle = False
        strategy_id = None

        # 1. 删除历史记录
        if history_file.exists():
            try:
                import json
                history = json.loads(history_file.read_text(encoding="utf-8"))
                # 找到对应记录，提取 strategy_id（如果有）
                for h in history:
                    if h.get("run_id") == run_id:
                        result = h.get("result") or {}
                        strategy_id = result.get("strategy_id")
                        break
                # 删除记录
                history = [h for h in history if h.get("run_id") != run_id]
                history_file.write_text(
                    json.dumps(history, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                deleted_history = True
            except Exception as exc:  # noqa: BLE001
                _logger.warning("删除历史记录失败: %s", exc)

        # 2. 删除 lifecycle 入库记录（如果有）
        if strategy_id:
            try:
                from ...knowledge import KnowledgeStore
                kb = KnowledgeStore()
                deleted_lifecycle = kb.delete_strategy_lifecycle(strategy_id)
            except Exception as exc:  # noqa: BLE001
                _logger.warning("删除 lifecycle 记录失败: %s", exc)

        return {
            "deleted_history": deleted_history,
            "deleted_lifecycle": deleted_lifecycle,
            "strategy_id": strategy_id,
        }

    def _promote_verified(self, req, per_symbol: list, generated_code: str,
                          out: Dict[str, Any], extra_note: str = "") -> None:
        """把 gate=verified 的品种写入 lifecycle（有效策略库），结果写回 out。"""
        verified_syms = list(dict.fromkeys(
            p["symbol"] for p in per_symbol
            if (p.get("gate") or {}).get("status") == "verified"))
        if not verified_syms:
            return
        try:
            from ...knowledge import KnowledgeStore

            kb = KnowledgeStore()
            sid = f"val_{datetime.now():%Y%m%d%H%M%S}"
            reason = "；".join(
                f"{p['symbol']}：{(p.get('gate') or {}).get('reason', '')}"
                for p in per_symbol if p.get("gate", {}).get("status") == "verified")
            if extra_note:
                reason = f"{reason}；{extra_note}" if reason else extra_note
            kb.upsert_strategy_lifecycle(
                sid, idea=req.idea or "", state="BACKTEST", source="validate",
                code=generated_code or "",
                code_safe=bool(generated_code),
                symbols=verified_syms, status="verified",
                reason=reason[:300], brief=reason[:300],
            )
            try:
                kb.push_strategy_transition(
                    sid, "IDEA", "BACKTEST", "策略思想测试门槛通过")
            except Exception:  # noqa: BLE001
                pass
            out["promoted"] = True
            out["strategy_id"] = sid
            out["promoted_symbols"] = verified_syms
        except Exception as exc:  # noqa: BLE001
            out["promote_error"] = str(exc)[:200]

    async def _validate_with_optimization(self, req, provider=None, progress=None) -> Dict[str, Any]:
        """带防过拟合防线的参数优化版策略验证。

        流程（三防线）：
          1. IS/OOS 时间切分：IS 段网格穷举；OOS 段只验证 IS 选出的 top-K 组合，
             绝不参与搜索；
          2. 参数高原：最优组合的邻域中位 IS Sharpe 须达标（尖峰=样本内噪声）；
          3. Deflated Sharpe：按试验数 N 折算后作为入库判据。
        OOS 指标（而非 IS 最优）过现有 gate 判定后入库。
        """
        from ...strategy import optimizer as opt
        from ...strategy.validation import (
            DEFAULT_PARAM_GRIDS,
            DEFAULT_SETTINGS,
        )
        from ...research.knowledge_loop import judge_strategy
        from ..schemas import OptimizationConfig

        def _prog(msg: str, cur: int = 0, tot: int = 0) -> None:
            if progress:
                progress(msg, cur, tot)

        raw_cfg = req.optimization
        cfg = (raw_cfg if isinstance(raw_cfg, OptimizationConfig)
               else OptimizationConfig(**(raw_cfg or {})))
        symbols = [x for x in (getattr(req, "symbols", None) or []) if x and x.strip()]
        if not symbols and getattr(req, "symbol", ""):
            symbols = [getattr(req, "symbol", "")]
        if not symbols:
            return {"error": "未指定标的（symbols 为空）"}

        # ------------------------------------------------- 1) 策略来源（与单次路径一致）
        generated_code = ""
        info: Dict[str, Any] = {}
        approved = (getattr(req, "code", "") or "").strip()
        if approved:
            # 用户在对话式编程阶段审定的代码：跳过 LLM，直接注册（仍过沙箱校验）
            code = approved
        else:
            code, err = await self._llm_generate_strategy(provider, req.idea, interval=req.interval)
            if err:
                return {"error": f"LLM 策略编程失败：{err}"}
        name = "idea_strategy"
        ok, err2, info = self.register_generated_strategy(name, code)
        if not ok:
            return {"error": f"策略注册失败：{err2}"}
        cls = self._extra_strategies.get(name)
        generated_code = code
        strategy_desc = "用户审定的 LLM 策略" if approved else "LLM 预编程策略"

        # ------------------------------------------------- 2) 参数网格解析
        # 优先级：请求显式 > 生成代码内 PARAM_GRID > 策略类自动推导 > 预置模板内置
        grid_source = "builtin"
        param_grid = {k: list(v) for k, v in (cfg.param_grid or {}).items() if v}
        if param_grid:
            grid_source = "request"
        if not param_grid:
            param_grid = {k: list(v) for k, v in
                          (info.get("param_grid") or {}).items()} if info else {}
            if param_grid:
                grid_source = "code"
        if not param_grid:
            param_grid = opt.auto_param_grid(cls, source=generated_code or None)
            if param_grid:
                grid_source = "auto"
        if not param_grid:
            param_grid = dict(DEFAULT_PARAM_GRIDS.get(name, {}))
        if not param_grid:
            return {"error": "未找到参数网格：请显式传 optimization.param_grid，"
                             "在生成代码中输出 PARAM_GRID，或给策略参数设置数值默认值"}
        try:
            # 多参数策略友好：笛卡尔积超上限时均匀削减（保留端点）而非直接报错
            param_grid = opt.fit_grid(param_grid, max_combos=cfg.max_combos)
            combos = opt.enumerate_grid(param_grid, max_combos=cfg.max_combos)
        except ValueError as exc:
            return {"error": f"参数网格非法：{exc}"}

        _valid_ivs = ("1d", "1h", "30m", "15m", "5m", "1m")
        intervals = [x for x in (getattr(req, "intervals", None) or []) if x in _valid_ivs]
        if not intervals:
            intervals = [req.interval if req.interval in _valid_ivs else "1d"]
        base_setting = dict(req.setting or {}) or dict(DEFAULT_SETTINGS.get(name) or {})

        per_symbol: list = []
        n_trials_total = 0
        is_bars_total = oos_bars_total = 0
        grid_used: Dict[str, list] = param_grid

        _total_steps = max(1, len(combos) * len(symbols) * len(intervals))
        _done_steps = 0

        for iv in intervals:
            for _i, sym in enumerate(symbols, 1):
                _prog(f"参数优化 {sym}@{iv}（{_i}/{len(symbols)}）：IS 段网格回测…",
                      _done_steps, _total_steps)
                vt = f"{sym}.{req.exchange.upper()}"
                try:
                    bars = await self.dm.get_bar_data(HistoryRequest(
                        symbol=sym,
                        exchange=Exchange(req.exchange.upper()),
                        interval=Interval(iv),
                        start=datetime.fromisoformat(req.start) if req.start else None,
                        end=datetime.fromisoformat(req.end) if req.end else None,
                    ))
                except Exception as exc:  # noqa: BLE001
                    per_symbol.append({"symbol": sym, "interval": iv,
                                       "error": f"数据获取失败：{exc}"})
                    continue
                if not bars:
                    per_symbol.append({"symbol": sym, "interval": iv,
                                       "error": "无数据（检查 data_cache）"})
                    continue
                if len(bars) < 200:
                    per_symbol.append({"symbol": sym, "interval": iv,
                                       "error": f"数据不足（{len(bars)} 根 < 200），无法切分"})
                    continue

                is_bars, oos_bars, split_info = opt.split_is_oos(
                    bars, is_ratio=cfg.is_ratio, warmup_bars=cfg.warmup_bars)
                if split_info["degraded"]:
                    per_symbol.append({"symbol": sym, "interval": iv,
                                       "error": f"历史过短（{len(bars)} 根），OOS 段不足，请拉长日期范围"})
                    continue

                # 周期-策略兼容性校验 + 多周期上下文（与主路径一致）
                if generated_code:
                    from ...backtest.interval_check import check_strategy_interval_compatibility
                    compat = check_strategy_interval_compatibility(generated_code, iv)
                    if not compat["compatible"]:
                        per_symbol.append({"symbol": sym, "interval": iv,
                                           "error": "策略与数据周期不兼容：" + "；".join(compat["issues"])})
                        continue
                daily_ctx = None
                mtf = None
                _needs_tf = generated_code and (
                    "self.daily" in generated_code or "self.mtf" in generated_code)
                if iv != "1d":
                    try:
                        _daily_bars = await self.dm.get_bar_data(HistoryRequest(
                            symbol=sym,
                            exchange=Exchange(req.exchange.upper()),
                            interval=Interval.DAILY,
                            start=None,
                            end=datetime.fromisoformat(req.end) if req.end else None,
                        ))
                        if _daily_bars:
                            from ...strategy.daily_context import DailyContext
                            from ...strategy.multi_tf import MultiTFContext, resample_bars
                            daily_ctx = DailyContext(_daily_bars)
                            mtf = MultiTFContext()
                            mtf.add("1d", _daily_bars,
                                    close_times=_daily_close_times(_daily_bars))
                            _wb, _wb_ct = resample_bars(_daily_bars, "1w", "1d")
                            if _wb:
                                mtf.add("1w", _wb, close_times=_wb_ct)
                    except Exception:  # noqa: BLE001 —— 日线上下文缺失走下方显式错误
                        _logger.exception("日线上下文构建失败 iv=%s", iv)
                        daily_ctx = None
                if _needs_tf and daily_ctx is None:
                    per_symbol.append({"symbol": sym, "interval": iv,
                                       "error": f"策略依赖日线级上下文（self.daily/self.mtf），但品种 {sym} 的日线数据不可用"})
                    continue
                if mtf is not None:
                    from ...strategy.multi_tf import RESAMPLE_CANDIDATES, resample_bars
                    for tgt in RESAMPLE_CANDIDATES.get(iv, []):
                        try:
                            _rb, _rct = resample_bars(bars, tgt, iv)
                            mtf.add(tgt, _rb, close_times=_rct)
                        except Exception:  # noqa: BLE001
                            pass

                sizes = {vt: default_size(vt)}

                # -------- IS 段网格穷举（逐组合同步线程池执行）
                is_results: Dict[tuple, Dict[str, Any]] = {}
                for _ci, combo in enumerate(combos, 1):
                    _done_steps += 1
                    _prog(f"参数优化 {sym}@{iv}：IS 组合 {_ci}/{len(combos)}（全任务 {_done_steps}/{_total_steps}）…",
                          _done_steps, _total_steps)
                    setting = {**base_setting, **combo}
                    r = await asyncio.to_thread(
                        run_strategy, "backtest", cls, vt, setting,
                        is_bars, self.ee, sizes, "ctp", None, req.cost,
                        daily_context=daily_ctx, mtf_context=mtf,
                    )
                    rep = r.get("report") or {}
                    is_results[opt.combo_key(combo)] = {
                        "combo": combo,
                        "sharpe": _safe_num(rep.get("sharpe")),
                        "trades": int(r.get("trades", 0) or 0),
                        "max_drawdown": _safe_num(rep.get("max_drawdown")),
                    }
                n_trials_total += len(is_results)

                # -------- 淘汰交易笔数不足的组合（Sharpe 无统计意义）
                valid = {k: v for k, v in is_results.items()
                         if v["trades"] >= cfg.min_trades}
                if not valid:
                    per_symbol.append({
                        "symbol": sym,
                        "interval": iv,
                        "error": f"所有 {len(is_results)} 个组合在 IS 段成交均 < "
                                 f"{cfg.min_trades} 笔，无法评估",
                    })
                    continue

                # -------- 取 IS Top-K → OOS 段各验证一次（绝不回头调参）
                # sharpe 可能为 None（报告缺失）：排序/取优时降序把 None 排尾，避免 TypeError
                ranked = sorted(
                    valid.values(),
                    key=lambda v: (v["sharpe"] is None, v["sharpe"] if v["sharpe"] is not None else 0.0),
                    reverse=True,
                )
                _prog(f"参数优化 {sym}@{iv}：样本外验证 top-{max(1, cfg.top_k)}…",
                      _done_steps, _total_steps)
                oos_runs = []
                for meta in ranked[: max(1, cfg.top_k)]:
                    setting = {**base_setting, **meta["combo"]}
                    r = await asyncio.to_thread(
                        run_strategy, "backtest", cls, vt, setting,
                        oos_bars, self.ee, sizes, "ctp", None, req.cost,
                        warmup_bars=split_info.get("warmup_bars", 0),
                        daily_context=daily_ctx, mtf_context=mtf,
                    )
                    rep = r.get("report") or {}
                    oos_runs.append({**meta, "report": rep, "raw": r})
                best = max(
                    oos_runs,
                    key=lambda x: (x["sharpe"] is None, x["sharpe"] if x["sharpe"] is not None else 0.0),
                )
                best_report = best["report"]

                # -------- 高原检验（基于 IS 段各组合 Sharpe）
                plateau = opt.plateau_check(
                    param_grid, best["combo"],
                    {k: v["sharpe"] for k, v in valid.items()},
                    ratio_threshold=cfg.plateau_ratio)

                # -------- DSR（按试验数 N 折算；试验 Sharpe 分布来自 IS 穷举）
                oos_returns = opt.daily_returns_from_equity(
                    best["raw"].get("equity_curve") or [])
                dsr = opt.deflated_sharpe(
                    best["sharpe"], oos_returns, n_trials_total,
                    trial_sharpes=[v["sharpe"] for v in valid.values()],
                ) if cfg.use_dsr else None

                item: Dict[str, Any] = {
                    "symbol": sym,
                    "interval": iv,
                    "exchange": req.exchange.upper(),
                    "bars": split_info["is_bars"] + split_info["oos_bars"],
                    "report": best_report,
                    "equity_curve": _sanitize(best["raw"].get("equity_curve") or []),
                    "trades": best["raw"].get("trades", 0),
                    "trade_list": _sanitize(best["raw"].get("trade_list") or []),
                    "optim_detail": {
                        "best_combo": best["combo"],
                        "is_sharpe": best["sharpe"],
                        "oos_sharpe": _safe_num(best_report.get("sharpe")),
                        "oos_trades": int(best["raw"].get("trades", 0) or 0),
                        "dsr": round(dsr, 4) if dsr is not None else None,
                        "plateau": plateau,
                        "top": [{"combo": t["combo"], "is_sharpe": t["sharpe"]}
                                for t in ranked[: max(1, cfg.top_k)]],
                    },
                }

                # -------- gate：判据用 OOS 指标 + DSR/高原附加拦截
                if req.gate:
                    gate = dict(req.gate)
                    sharpe = _safe_num(best_report.get("sharpe"))
                    mdd = _safe_num(best_report.get("max_drawdown"))
                    total_ret = _safe_num(best_report.get("total_return"))
                    total_cost = _safe_num(best_report.get("total_cost"))
                    cost_ratio = _safe_num(best_report.get("cost_ratio"))
                    judge = await judge_strategy(
                        None, {"run_id": vt, "state": "BACKTEST", "status": "",
                               "sharpe": sharpe, "max_drawdown": mdd},
                        gate=gate, fallback_rules=True)
                    status = judge.get("status")
                    reason = judge.get("reason", "")
                    if cfg.use_dsr and dsr is not None and status == "verified" and dsr < 0.9:
                        status = "rejected"
                        reason = (reason + "；" if reason else "") + (
                            f"Deflated Sharpe {dsr:.2f} < 0.9（试验 {n_trials_total} 次，"
                            "选择偏差校正后不可信）")
                    if not plateau["ok"] and status == "verified":
                        status = "rejected"
                        reason = (reason + "；" if reason else "") + (
                            "参数落在尖峰而非高原（" + plateau.get("reason", "") + "）")
                    max_cost_ratio = gate.get("max_cost_ratio", 0.6) or 0.6
                    if req.cost and max_cost_ratio > 0 and cost_ratio > max_cost_ratio:
                        status = "rejected"
                        reason = (reason + "；" if reason else "") + (
                            f"成本/净收益 {cost_ratio:.1%} 超上限 {max_cost_ratio:.0%}"
                            f"（高换手，总成本 {total_cost:.0f}）")
                    item["gate"] = {
                        "enabled": True,
                        "status": status,
                        "reason": reason[:300],
                        "tags": judge.get("tags") or [],
                        "metrics": {"sharpe": sharpe, "max_drawdown": mdd,
                                    "total_return": total_ret,
                                    "total_cost": total_cost, "cost_ratio": cost_ratio,
                                    "dsr": round(dsr, 4) if dsr is not None else None},
                    }
                per_symbol.append(item)
                is_bars_total += split_info["is_bars"]
                oos_bars_total += split_info["oos_bars"]

        out: Dict[str, Any] = {
            "idea": req.idea or "",
            "strategy": name,
            "strategy_desc": strategy_desc,
            "code": generated_code,
            "interval": "+".join(intervals),
            "per_symbol": per_symbol,
            "gate_enabled": bool(req.gate),
            "optim": {
                "enabled": True,
                "n_trials": n_trials_total,
                "is_bars": is_bars_total,
                "oos_bars": oos_bars_total,
                "param_grid": param_grid,
                "grid_source": grid_source,
                "is_ratio": cfg.is_ratio,
                "top_k": cfg.top_k,
                "use_dsr": cfg.use_dsr,
            },
        }

        # -------- 达标品种入库（判据已在逐品种 gate 中折入 DSR/高原）
        if req.gate and req.promote:
            self._promote_verified(
                req, per_symbol, generated_code, out,
                extra_note=(f"参数网格优化：{n_trials_total} 次试验（IS/OOS="
                            f"{cfg.is_ratio:.0%}/" f"{1 - cfg.is_ratio:.0%}），"
                            "OOS 指标过门槛"))
        return out

    async def _llm_generate_strategy(self, provider, idea: str,
                                     history: Optional[List[Dict[str, str]]] = None,
                                     interval: str = "1d") -> "tuple[str, str]":
        """LLM 预编程：策略思想 → CtaTemplate 策略代码（含沙箱自修复循环）。

        委托给 :meth:`draft_strategy_code`：沙箱/周期兼容性校验失败时自动把
        真实错误喂回 LLM 自行修订（最多 ``_DRAFT_REPAIR_ROUNDS`` 轮），
        全部轮次失败仍返回末版代码供人工修复。
        返回 (code, err)；err 非空表示失败（失败闭合，不抛异常）。
        """
        out = await self.draft_strategy_code(provider, idea, history=history,
                                             interval=interval)
        if out.get("error"):
            return "", out["error"]
        if not out.get("sandbox_ok"):
            return "", (f"沙箱校验未通过（LLM 自修复 {out.get('repair_rounds', 0)} 轮未成功）："
                        f"{out.get('sandbox_err') or '未知原因'}。"
                        "末版代码已保留在编辑器中，可手动修正或填写修改意见重试。")
        return out.get("code") or "", ""


    async def draft_strategy_code(self, provider, idea: str,
                                  history: Optional[List[Dict[str, str]]] = None,
                                  interval: str = "1d") -> Dict[str, Any]:
        """对话式策略编程草稿：生成/修改策略代码，**不回测**。

        自修复循环（借鉴 self-healing generation）：沙箱/周期兼容性校验失败时，
        把真实错误喂回 LLM 自行修订（最多 ``_DRAFT_REPAIR_ROUNDS`` 轮），
        通过后才返回给用户；全部轮次失败仍返回末版代码与错误详情，
        供用户在界面上审阅手动修改。
        返回 {code, sandbox_ok, sandbox_err, repair_rounds, provider} 或 {error}。
        """
        if provider is None:
            return {"error": "LLM Provider 不可用（请先配置 AI Key）"}
        if getattr(provider, "name", "") == "mock":
            return {"error": ("当前为 Mock Provider（未配置 AI Key）：生成的是占位演示代码，"
                              "不能用于回测。请先在「设置」页或 .env 配置 QM_LLM_* 后重试。")}
        from ...ai.sandbox import compile_strategy
        from ...backtest.interval_check import check_strategy_interval_compatibility

        msgs = list(history or [])
        if idea and not any(m.get("role") == "user" for m in msgs):
            msgs.append({"role": "user", "content": str(idea)})
        if not msgs:
            return {"error": "策略思想为空"}
        # 系统提示词与正式生成一致；历史截断防爆 token
        try:
            if len(msgs) > 8:
                msgs = msgs[-8:]
        except Exception:  # noqa: BLE001
            pass
        # 周期信息：告知当前周期及多周期上下文可用性（按周期动态生成）
        if interval == "1d":
            _hint = ("\n\n【数据周期】1d\n"
                     "- 本次数据周期为日线（1d）：self.mtf/self.daily 不可用，"
                     "禁止在代码中引用\n")
        else:
            from ...strategy.multi_tf import RESAMPLE_CANDIDATES
            _tfs = sorted(set(["1d", "1w"] + RESAMPLE_CANDIDATES.get(interval, [])))
            _hint = (f"\n\n【数据周期】{interval}（分钟级）\n"
                     "- 日内策略的时间判断逻辑（hour/minute）在分钟数据上正常触发\n"
                     f"- 框架已注入多周期上下文 self.mtf，可用周期：{', '.join(_tfs)}\n"
                     "- 更高周期规则（如日线定方向、前日高低点、周线趋势）优先用 self.mtf 实现\n"
                     "- 指标数据深度不足时返回 None（预热期），代码需判空跳过\n")
        if msgs and msgs[-1].get("role") == "user":
            msgs[-1]["content"] = msgs[-1]["content"] + _hint
        else:
            msgs.append({"role": "user", "content": _hint})

        system = _strategy_codegen_system()
        code, sandbox_ok, sandbox_err = "", False, ""
        repair_rounds = 0
        # 自修复循环：首次生成 + N 轮沙箱失败自动修订
        for attempt in range(_DRAFT_REPAIR_ROUNDS + 1):
            try:
                code = await provider.chat_messages(system, msgs)
            except Exception as exc:  # noqa: BLE001
                return {"error": f"LLM 调用失败：{exc}"}
            # 真实 Provider 失败会静默回退 Mock（返回占位代码）——此处转为显式错误
            fallback = getattr(provider, "last_fallback_reason", None)
            if fallback:
                return {"error": f"真实 LLM 调用失败（返回了 Mock 占位代码）：{fallback}"}
            code = _strip_code_fences(code or "").strip()
            if not code:
                return {"error": "LLM 未返回代码"}
            ok, err, _ = compile_strategy(code, require_base="CtaTemplate")
            compat = check_strategy_interval_compatibility(code, interval)
            sandbox_ok = bool(ok and compat["compatible"])
            sandbox_err = (err or "") if not ok else (
                "；".join(compat["issues"]) if not compat["compatible"] else "")
            if sandbox_ok:
                break
            if attempt < _DRAFT_REPAIR_ROUNDS:
                _problems = []
                if not ok:
                    _problems.append(f"沙箱校验失败：{err}")
                if not compat["compatible"]:
                    _problems.append("策略与数据周期不兼容：" + "；".join(compat["issues"]))
                repair_rounds = attempt + 1  # 已发起的修复轮数
                msgs = msgs + [
                    {"role": "assistant", "content": code},
                    {"role": "user", "content": (
                        "以上代码未通过校验：" + "；".join(_problems)
                        + "。请修复全部问题后重新输出完整代码（只输出代码本身，不要解释）。")},
                ]
        return {"code": code, "sandbox_ok": sandbox_ok,
                "sandbox_err": sandbox_err if not sandbox_ok else "",
                "repair_rounds": repair_rounds,
                "provider": getattr(provider, "name", "")}

    async def run_walkforward(self, req: WalkForwardRequest) -> Dict[str, Any]:
        """Walk-Forward 滚动样本外验证"""
        strat_class = self._resolve_strategy_class(req.strategy)
        if strat_class is None:
            return {"error": f"未知策略: {req.strategy}"}
        vt = f"{req.symbol}.{req.exchange.upper()}"

        min_bars_needed = req.train_window + req.test_window * 2
        days_needed = max(min_bars_needed + 50, 500)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_needed)

        bars = await self.dm.get_bar_data(
            HistoryRequest(
                symbol=req.symbol,
                exchange=Exchange(req.exchange.upper()),
                interval=Interval("1d"),
                start=start_date,
                end=end_date,
            )
        )
        if not bars:
            return {"error": "无数据"}

        min_required = req.train_window + req.test_window
        if len(bars) < min_required:
            return {
                "error": f"样本不足：需要至少 {min_required} 根，当前仅 {len(bars)} 根。"
                f"请减小 train_window/test_window。"
            }

        sizes = {vt: default_size(vt)}
        try:
            result = await asyncio.to_thread(
                walk_forward,
                bars,
                strat_class,
                dict(req.setting),
                vt,
                req.train_window,
                req.test_window,
                req.step,
                sizes,
                req.capital,
                req.cost if req.cost else None,
            )
            return _sanitize(result.to_dict())
        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"运行失败: {str(e)}"}

    async def run_paper(self, req: "PaperRunRequest") -> Dict[str, Any]:
        """模拟盘实跑：把策略（含 AI 注册策略）部署到 PaperEngine 做历史回放。

        复用 ``run_strategy(mode="paper")`` 的同一套策略代码与撮合模型，
        产出与实盘同源的 PaperEngine 回放结果（现金/持仓/成交/权益），
        供生命周期晋升到 PAPER 及前端「模拟盘实跑」闭环使用。
        """
        strat_class = self._resolve_strategy_class(req.strategy)
        if strat_class is None:
            return {"error": f"策略不存在或未注册: {req.strategy}"}
        vt = f"{req.symbol}.{req.exchange.upper()}"

        # 发送开始事件
        await ws_manager.broadcast({
            "type": "backtest_start",
            "strategy": req.strategy,
            "symbol": req.symbol,
            "exchange": req.exchange,
            "mode": "paper",
            "timestamp": datetime.now().isoformat(),
        })

        end_date = datetime.now()
        start_date = end_date - timedelta(days=req.days or 400)
        try:
            bars = await self.dm.get_bar_data(
                HistoryRequest(
                    symbol=req.symbol,
                    exchange=Exchange(req.exchange.upper()),
                    interval=Interval("1d"),
                    start=start_date,
                    end=end_date,
                )
            )
        except Exception as exc:  # noqa: BLE001
            await ws_manager.broadcast({
                "type": "backtest_error",
                "strategy": req.strategy,
                "error": f"取数失败: {exc}",
            })
            return {"error": f"取数失败: {exc}"}
        if not bars:
            await ws_manager.broadcast({
                "type": "backtest_error",
                "strategy": req.strategy,
                "error": "无数据",
            })
            return {"error": "无数据"}

        # 发送进度事件
        await ws_manager.broadcast({
            "type": "backtest_progress",
            "strategy": req.strategy,
            "progress": 0.3,
            "message": f"已加载 {len(bars)} 根K线",
        })

        sizes = {vt: default_size(vt)}
        try:
            result = await asyncio.to_thread(
                run_strategy,
                "paper",
                strat_class,
                vt,
                dict(req.setting),
                bars,
                self.ee,
                sizes,
                "ctp",
                None,
                None,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.error(f"模拟盘实跑失败: {req.strategy} on {vt} - {exc}", exc_info=True)
            await ws_manager.broadcast({
                "type": "backtest_error",
                "strategy": req.strategy,
                "error": f"模拟盘实跑失败: {exc}",
            })
            return {"error": f"模拟盘实跑失败: {exc}"}

        summary = result.get("summary", {})
        # 用权益曲线算真实夏普/回撤（参考 validation.py 年化 252 公式）；缺权益则回退纯纸面
        _eq = _extract_equity_curve(result)
        _eq_metrics = _metrics_from_equity(_eq) if _eq is not None else {}
        out = {
            "ok": True,
            "strategy": req.strategy,
            "vt_symbol": vt,
            "bars": len(bars),
            "trade_count": result.get("trades", 0),
            "cash": summary.get("cash"),
            "positions": summary.get("positions", {}),
            "metrics": {
                "trade_count": result.get("trades", 0),
                "final_cash": summary.get("cash"),
                "open_positions": len(summary.get("positions", {})),
                "sharpe": _eq_metrics.get("sharpe"),
                "max_drawdown": _eq_metrics.get("max_drawdown"),
                "total_return": _eq_metrics.get("total_return"),
            },
        }
        
        # 发送完成事件
        await ws_manager.broadcast({
            "type": "backtest_complete",
            "strategy": req.strategy,
            "progress": 1.0,
            "message": "模拟盘实跑完成",
            "summary": summary,
        })
        
        return _sanitize(out)


def _extract_equity_curve(result: Dict[str, Any]) -> pd.Series:
    """从回测结果中提取权益曲线（归一化为数值 Series）。

    equity_curve 可能为 ``[{date, equity...}, ...]`` 的 dict 列表，
    也可能为纯数值列表；这里统一转成 float Series，供指标计算使用。
    """
    raw = None
    if "equity_curve" in result:
        raw = result["equity_curve"]
    elif "portfolio" in result and "equity" in result["portfolio"]:
        raw = result["portfolio"]["equity"]
    elif "summary" in result and "equity_curve" in result["summary"]:
        raw = result["summary"]["equity_curve"]
    if raw is None:
        return pd.Series(dtype=float)

    series = pd.Series(raw)
    # 若元素为 dict，抽取数值字段；否则按原值处理
    if series.notna().any() and isinstance(series.dropna().iloc[0], dict):
        first = series.dropna().iloc[0]
        key = next((k for k in ("equity", "value", "balance", "nav") if k in first), None)
        if key is not None:
            series = series.apply(lambda d: d.get(key) if isinstance(d, dict) else d)
    return pd.to_numeric(series, errors="coerce")

def _metrics_from_equity(equity_curve):
    """从权益曲线算出年化夏普(252)/最大回撤/总收益（纯 pandas，无新依赖）。

    参考 backtest/validation.py 年化公式：mean/std * sqrt(252)。
    权益曲线过短或全为 NaN 时返回空 dict（调用方回退纸面判读）。
    """
    if equity_curve is None or getattr(equity_curve, "empty", True):
        return {}
    s = pd.Series(equity_curve).dropna()
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 2 or float(s.iloc[0]) == 0:
        return {}
    rets = s.pct_change().dropna()
    std = rets.std()
    sharpe = float((rets.mean() / std * (252 ** 0.5))) if std and std > 0 else 0.0
    running_max = s.cummax()
    dd = s / running_max - 1.0
    mdd = float(dd.min())
    total = float((s.iloc[-1] / s.iloc[0]) - 1.0)
    return {"sharpe": sharpe, "max_drawdown": mdd, "total_return": total}
