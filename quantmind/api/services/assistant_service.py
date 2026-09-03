"""AI 投资助手：常驻对话服务（参考主观投资-docker「AI 修正助手」的 agent 循环）。

- **工具循环**：LLM 通过 JSON 动作协议调用工具（读回测/校验代码/查知识库/
  列策略库），后端执行后把结果喂回下一轮，直到给出最终自然语言回复；
  每次工具调用/结果都通过 progress events 透出（前端渲染🔧工具行+↳结果摘要）。
- **系统提示词可编辑**：支持自定义持久化，并可让 LLM 扫描平台知识自动再生成。
- **会话持久化**：对话历史存 data_cache/assistant_session.json，可归档新开对话。
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_logger = logging.getLogger("quantmind.api.assistant")

_SESSION_FILE = Path("data_cache") / "assistant_session.json"
_ARCHIVE_FILE = Path("data_cache") / "assistant_session_archive.json"
_PROMPT_FILE = Path("data_cache") / "assistant_system_prompt.json"
_RUNS_FILE = Path("data_cache") / "strategy_validation_runs.json"

# 对话历史上限（条数，user+assistant 合计），超出截断最早的
_MAX_MESSAGES = 60
# 工具循环上限（防止死循环烧 token）
_MAX_TOOL_ROUNDS = 4
# 单次传回 LLM 的工具结果上限（字符）
_TOOL_RESULT_LIMIT = 2000

# ---------------------------------------------------------------------------
# 系统提示词
# ---------------------------------------------------------------------------

_SYSTEM_BASE = """你是 QuantMind 量化投研平台的「AI 投资助手」，运行在网页右下角的对话面板中。

【平台背景】
- 平台核心工作流：策略思想（自然语言）→ LLM 预编程为 CtaTemplate 策略代码 → AST 沙箱校验 → 多品种逐个真实数据回测 → 门槛判定（Sharpe/回撤/成本占比）→ 达标入有效策略库。
- 策略代码是 vnpy 风格的 CtaTemplate 子类：构造参数经 `parameters` 列表声明，`on_init` 预热，`on_bar(bar)` 处理 K 线；下单用 self.buy/sell/short/cover，撤单用 self.cancel_all()。分钟级周期有 self.mtf 多周期上下文。
- 回测引擎：vnpy_backtest，支持多品种逐个回测，成本模型包含手续费/平今/印花税/滑点。
- 数据源：AKShare（A 股/期货）、efinance（期货）、mootdx（期货）、yfinance（美股/港股）、本地 Parquet 缓存。数据质量可能因源而异，AKShare 偶尔延迟，efinance/mootdx 较稳定。
- LLM 提供商：DeepSeek（默认）/OpenAI/Anthropic/Qwen，通过标准 OpenAI 接口调用，支持 mock 模式（无 Key 时）。
- AST 沙箱校验：禁止网络请求/文件 IO/动态执行（eval/exec/import 非白名单模块），只允许 CtaTemplate 子类及基础库（numpy/pandas/ta-lib）。
- 回测指标：total_return 总收益、annual_return 年化、sharpe、max_drawdown 最大回撤、win_rate 胜率、profit_factor 盈亏比、cost_ratio 成本占净收益比。
- 可靠性口径：回测覆盖 <60 个交易日或 <20 笔交易时，年化等指标只是噪声；前 3 大盈利日贡献占比过高说明收益依赖个别行情日；OOS/IS 保持度低、DSR<0.9、参数尖峰都说明过拟合。
- 参数优化：IS/OOS 时间切分（默认 70/30），网格只在 IS 搜索，OOS 验证 Top-K；加 DSR 多重检验校正 + 参数高原检验，防止过拟合。
- 支持品种：IC0/IF0/IH0/IM0（中金所股指）、rb0/cu0/au0/ag0（上期所）、m0/ta0/i0/j0/y0/SR0/CF0/MA0（郑商所/大商所）等期货；A 股/港股/美股（通过 yfinance）。
- K 线周期：1d（日线）、1h、30m、15m、5m、1m（分钟级）；日线策略不可用 self.mtf，分钟级策略优先用 self.mtf 实现多周期逻辑。

【你的职责】
- 用简洁中文回答；解释代码时逐条说明入场/离场/止损规则对应的代码位置。
- 解读回测结果时，主动引用样本量、利润集中度、OOS 衰减等可靠性证据，不做没有依据的乐观结论。
- 用户要求修改代码时：给出**完整可运行的全量代码**（单个 ```python 代码块），界面会提供「应用到策略代码区」按钮；修改前列出改动点。
- 用户的问题超出量化投研范围时，简短说明并引导回主题。

【可用工具（JSON 动作协议）】
需要查数据/校验代码时，**只回复一个 JSON 对象**（不要包裹多余文本）：
{{"reply": "一句话说明你正在做什么", "actions": [{{"tool": "工具名", "args": {{}}}}]}}
可用工具：
- read_backtest —— 读取最近一次「LLM 策略挖掘」多品种回测的完整结果。args: {{"symbol": "可选，只看某品种，如 IC0"}}
- validate_code —— 对策略代码做 AST 沙箱校验（修改代码后建议自检）。args: {{"code": "完整策略代码"}}
- search_knowledge —— 检索平台知识库（因子/策略/研究日志/方法论）。args: {{"query": "关键词", "top_k": 5}}
- list_strategies —— 列出策略生命周期/有效策略库最新状态。args: {{"limit": 10}}
- apply_code —— 直接应用策略代码到界面（沙箱校验通过后自动更新）。args: {{"code": "完整策略代码"}}
不需要工具时，直接用自然语言回复，**不要输出 JSON**。

【工具调用纪律】
- 用户问及回测结果/数据，而上下文中没有具体数据时，**必须先调用 read_backtest 获取**，禁止让用户手工粘贴平台已有的数据。
- 用户要求修改代码时，修改后应调用 apply_code 直接应用（内置沙箱校验），不要只给代码让用户手动复制。
- 涉及平台已有策略/因子/方法论事实时，用 search_knowledge / list_strategies 查证，不要凭空编造。
"""

_SYSTEM_CONTEXT_TMPL = """
【当前用户工作流上下文】
{context}
"""

_REGENERATE_INSTRUCTION = """请为 QuantMind 量化投研平台重新生成一份「AI 投资助手」系统提示词。
平台背景：Python 3.13 + Streamlit 前端 + FastAPI 后端；核心页面包括仪表盘、行情数据、
因子研究、LLM策略挖掘（策略思想→LLM预编程→沙箱校验→多品种回测→门槛→策略库）、
策略回测、Walk-Forward、风控中心、生命周期管理等。
要求：1) 保留助手职责说明（解释代码、解读回测、修改策略、引导回量化主题）；
2) 包含回测指标口径与可靠性判据（样本量、利润集中度、OOS 衰减、DSR）；
3) 包含 CtaTemplate 策略代码规范；4) 中文、结构化、800 字以内。
直接输出生成的提示词全文，不要解释。"""


def _load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            _logger.warning("读取 %s 失败: %s", path.name, e)
    return default


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 会话管理
# ---------------------------------------------------------------------------

def load_session() -> Dict[str, Any]:
    """读取助手会话（消息列表）。"""
    data = _load_json(_SESSION_FILE, None)
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        return data
    return {"messages": []}


def clear_session() -> None:
    """清空助手会话（删除聊天记录）。"""
    try:
        if _SESSION_FILE.exists():
            _SESSION_FILE.unlink()
    except Exception as e:  # noqa: BLE001
        _logger.warning("清空助手会话失败: %s", e)


def new_session() -> None:
    """归档当前会话并新开（「🆕 新建对话」）。"""
    if _SESSION_FILE.exists():
        try:
            archive = _load_json(_ARCHIVE_FILE, [])
            if not isinstance(archive := _load_json(_ARCHIVE_FILE, []), list):
                archive = []
            entry = _load_json(_SESSION_FILE, {})
            entry["archived_at"] = datetime.now().isoformat(timespec="seconds")
            archive.insert(0, entry)
            _write_json(_ARCHIVE_FILE, archive[:10])
        except Exception as e:  # noqa: BLE001
            _logger.warning("归档会话失败: %s", e)
    clear_session()


def _save_messages(messages: List[Dict[str, str]]) -> None:
    try:
        _SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        _write_json(_SESSION_FILE, {
            "messages": messages[-60:],
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        })
    except Exception as e:  # noqa: BLE001
        _logger.warning("保存助手会话失败: %s", e)


# ---------------------------------------------------------------------------
# 系统提示词（可自定义 + AI 再生成）
# ---------------------------------------------------------------------------

def get_system_prompt() -> str:
    data = _load_json(_PROMPT_FILE, None)
    if isinstance(data, dict) and data.get("content"):
        return data["content"]
    return _SYSTEM_BASE


def set_system_prompt(content: str) -> None:
    _write_json(_PROMPT_FILE, {
        "content": content,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    })


def reset_system_prompt() -> None:
    try:
        if _PROMPT_FILE.exists():
            _PROMPT_FILE.unlink()
    except Exception as e:  # noqa: BLE001
        _logger.warning("重置系统提示词失败: %s", e)


def build_system_prompt(context: Optional[Dict[str, str]]) -> str:
    """系统提示词（自定义或内置）+ 当前工作流上下文。"""
    parts: List[str] = [get_system_prompt()]
    ctx_lines: List[str] = []
    if context:
        if context.get("idea"):
            ctx_lines.append(f"【策略思想】{context['idea']}")
        if context.get("code"):
            ctx_lines.append(f"【当前策略代码】\n```python\n{context['code']}\n```")
        if context.get("result_summary"):
            ctx_lines.append(f"【最近回测结果摘要】\n{context['result_summary']}")
    if ctx_lines:
        parts.append(_SYSTEM_CONTEXT_TMPL.format(context="\n".join(ctx_lines)))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 工具实现（agent 循环可调用）
# ---------------------------------------------------------------------------

def _tool_read_backtest(symbol: str = "") -> str:
    runs = _load_json(_RUNS_FILE, [])
    if not isinstance(runs, list) or not runs:
        return "暂无回测历史记录。"
    run = runs[0]  # 最新一次
    result = run.get("result") or {}
    lines = [f"运行时间 {run.get('created_at')} · 思想 {(run.get('idea') or '')[:80]}"]
    for p in result.get("per_symbol") or []:
        if symbol and p.get("symbol") != symbol:
            continue
        if p.get("error"):
            lines.append(f"- {p['symbol']}: 回测失败 {str(p['error'])[:80]}")
            continue
        r = p.get("report") or {}
        g = p.get("gate") or {}
        eq = p.get("equity_curve") or []
        days = len({str(e.get("date", ""))[:10] for e in eq})
        lines.append(
            f"- {p['symbol']}·{p.get('interval') or '1d'}: 总收益 {r.get('total_return', 0):+.2%}, "
            f"年化 {r.get('annual_return', 0):+.2%}, Sharpe {r.get('sharpe', 0):.2f}, "
            f"回撤 {r.get('max_drawdown', 0):.2%}, 胜率 {r.get('win_rate', 0):.0%}, "
            f"盈亏比 {r.get('profit_factor', 0):.2f}, {p.get('trades', 0)}笔/{days}交易日, "
            f"成本占比 {r.get('cost_ratio', 0):.1%}, 门槛 {g.get('status', '-')}")
    if result.get("optim"):
        lines.append(f"参数优化: {json.dumps(result['optim'], ensure_ascii=False)[:300]}")
    text = "\n".join(lines)
    return text[:3000]


def _tool_validate_code(code: str) -> str:
    if not (code or "").strip():
        return "代码为空，无法校验。"
    from ...ai.sandbox import compile_strategy
    ok, err, _ = compile_strategy(code, require_base="CtaTemplate")
    return "✅ 沙箱校验通过" if ok else f"❌ 沙箱校验失败：{err}"


def _tool_search_knowledge(query: str, top_k: int = 5) -> str:
    from ...knowledge import KnowledgeStore
    items = KnowledgeStore().search(query, top_k=max(1, min(10, int(top_k))))
    if not items:
        return f"知识库中未找到与「{query}」相关的内容。"
    lines = []
    for it in items:
        lines.append(f"- [{it.get('kind')}] {str(it.get('text', ''))[:120]}")
    return "\n".join(lines)[:2000]


def _tool_list_strategies(limit: int = 10) -> str:
    from ...knowledge import KnowledgeStore
    rows = KnowledgeStore().list_strategy_lifecycles(limit=max(1, min(50, int(limit))))
    if not rows:
        return "策略生命周期为空。"
    lines = []
    for r in rows[:limit]:
        lines.append(f"- {r.get('strategy_id')}: state={r.get('state')}, "
                     f"sharpe={r.get('sharpe')}, mdd={r.get('max_drawdown')}, "
                     f"status={r.get('status')}, reason={str(r.get('reason') or '')[:60]}")
    return "\n".join(lines)[:2000]


def _tool_apply_code(code: str) -> str:
    """直接应用代码到策略代码区（沙箱校验通过后自动更新）。"""
    if not (code or "").strip():
        return "❌ 代码为空"
    from ...ai.sandbox import compile_strategy
    ok, err, _ = compile_strategy(code, require_base="CtaTemplate")
    if not ok:
        return f"❌ 沙箱校验失败：{err}"
    # 写入待应用代码文件，前端轮询检测
    _PENDING_FILE = Path("data_cache") / "pending_code.json"
    _PENDING_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PENDING_FILE.write_text(json.dumps({
        "code": code,
        "applied_at": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False), encoding="utf-8")
    return "✅ 代码已通过沙箱校验并应用，界面已自动更新"


_TOOLS = {
    "read_backtest": _tool_read_backtest,
    "validate_code": _tool_validate_code,
    "search_knowledge": _tool_search_knowledge,
    "list_strategies": _tool_list_strategies,
    "apply_code": _tool_apply_code,
}


def _execute_tool(name: str, args: Dict[str, Any]) -> str:
    fn = _TOOLS.get(name)
    if fn is None:
        return f"未知工具：{name}（可用：{', '.join(_TOOLS)}）"
    try:
        return fn(**args)
    except Exception as e:  # noqa: BLE001
        _logger.exception("工具 %s 执行失败", name)
        return f"工具执行失败：{e}"


def _parse_action_json(raw: str) -> Optional[Dict[str, Any]]:
    """从 LLM 回复中提取动作 JSON（兼容 ```json 包裹）。"""
    m = re.search(r"\{[\s\S]*\}", raw or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None
    if isinstance(obj, dict) and isinstance(obj.get("actions"), list):
        return obj
    return None


# ---------------------------------------------------------------------------
# 主对话入口（agent 工具循环）
# ---------------------------------------------------------------------------

async def chat(provider: Any, message: str,
               history: Optional[List[Dict[str, str]]] = None,
               context: Optional[Dict[str, str]] = None,
               progress: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """助手对话（agent 循环）：LLM ↔ 工具 ↔ LLM，最终给出自然语言回复。

    progress（可选）：后台任务的可变 progress 字典，工具调用事件实时写入
    ``progress["events"]``，前端轮询 status 时渲染（🔧工具调用 / ↳结果摘要）。
    返回 {"reply": str, "events": [...]} 或 {"error": str}。
    """
    if provider is None:
        return {"error": "LLM Provider 不可用（请先配置 AI Key）"}
    if getattr(provider, "name", "") == "mock":
        return {"error": "当前为 Mock Provider（未配置 AI Key）。"
                         "请先在「设置」页或 .env 配置 QM_LLM_* 后再使用 AI 助手。"}

    events: List[Dict[str, Any]] = []

    def emit(ev: Dict[str, Any]) -> None:
        events.append(ev)
        if progress is not None:
            progress["events"] = list(events)

    msgs = [m for m in (history or []) if m.get("role") in ("user", "assistant")]
    msgs = msgs[-40:]
    if message.strip():
        msgs.append({"role": "user", "content": message.strip()})
    if not msgs:
        return {"error": "消息为空"}

    system = get_system_prompt()
    context_block = []
    if context:
        if context.get("idea"):
            context_block.append(f"【策略思想】{context['idea']}")
        if context.get("code"):
            context_block.append(f"【当前策略代码】\n```python\n{context['code']}\n```")
        if context.get("result_summary"):
            context_block.append(f"【最近回测结果摘要】\n{context['result_summary']}")
    if context_block:
        system += "\n" + _SYSTEM_CONTEXT_TMPL.format(context="\n".join(context_block))

    reply = ""
    base_msgs = list(msgs)  # 干净的对话史（不含工具内部轮次），持久化用
    for round_i in range(_MAX_TOOL_ROUNDS):
        try:
            raw = await provider.chat_messages(system, msgs)
        except Exception as exc:  # noqa: BLE001
            _logger.exception("助手 LLM 调用失败")
            return {"error": f"LLM 调用失败：{exc}"}
        # 真实 Provider 失败会静默回退 Mock（返回占位文本）——转为显式错误
        fallback = getattr(provider, "last_fallback_reason", None)
        if fallback:
            return {"error": f"真实 LLM 调用失败（返回了 Mock 占位回复）：{fallback}"}

        obj = _parse_action_json(raw)
        if obj is None:
            reply = raw.strip()
            break
        reply = str(obj.get("reply") or "")
        actions = obj.get("actions") or []
        if not actions:
            reply = str(obj.get("reply") or "").strip()
            break
        emit({"type": "text", "content": reply})
        msgs.append({"role": "assistant", "content": raw})
        results = []
        for a in actions[:4]:
            tool = str(a.get("tool") or "")
            args = a.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            emit({"type": "tool_call", "name": tool, "input": args})
            result = _execute_tool(tool, args)
            emit({"type": "tool_result", "name": tool,
                  "result": result[:500]})
            results.append(f"[工具 {tool} 结果]\n{result[:_TOOL_RESULT_LIMIT]}")
        msgs.append({"role": "user",
                     "content": "工具执行结果：\n\n" + "\n\n".join(results)
                                + "\n\n请基于以上结果继续。如已足够回答，直接给出最终回复"
                                  "（不要再调用工具，也不要输出 JSON）。"})
    else:
        if not reply:
            return {"error": f"工具循环达到上限（{_MAX_TOOL_ROUNDS} 轮），未得到最终回复"}

    if not reply:
        return {"error": "LLM 返回了空回复"}

    saved = [m for m in base_msgs if m.get("content")]
    saved.append({"role": "assistant", "content": reply})
    _save_messages(saved)
    return {"reply": reply, "events": events}


async def regenerate_system_prompt(provider: Any) -> Dict[str, Any]:
    """让 LLM 基于平台知识重新生成系统提示词（⚙️「AI 更新」）。"""
    if provider is None:
        return {"error": "LLM Provider 不可用（请先配置 AI Key）"}
    if getattr(provider, "name", "") == "mock":
        return {"error": "当前为 Mock Provider（未配置 AI Key），无法自动生成。"}
    instruction = (
        _SYSTEM_BASE + "\n\n【任务】\n" + _REGENERATE_INSTRUCTION
        + "\n\n【项目结构参考】\n"
        + json.dumps({
            "web_pages": ["仪表盘", "行情数据", "因子研究", "策略回测", "参数优化",
                          "风控中心", "WalkForward", "因子库", "知识库", "端到端流水线",
                          "因子组合策略", "LLM策略挖掘", "生命周期", "实时监控"],
            "ai_modules": ["ai/sandbox.py(AST沙箱)", "ai/codegen.py(代码生成)",
                           "strategy_mining/(架构师/编译器)", "research/(流水线)"],
            "backtest": ["多品种逐个回测", "门槛判定(Sharpe/回撤/成本占比)",
                         "网格参数优化(IS/OOS+DSR+参数高原)"],
        }, ensure_ascii=False))
    try:
        content = await provider.chat_messages(
            "你负责撰写量化投研平台的 AI 助手系统提示词。", [{"role": "user", "content": instruction}])
    except Exception as exc:  # noqa: BLE001
        return {"error": f"LLM 调用失败：{exc}"}
    fallback = getattr(provider, "last_fallback_reason", None)
    if fallback:
        return {"error": f"真实 LLM 调用失败：{fallback}"}
    set_system_prompt(content)
    return {"content": content, "length": len(content)}
