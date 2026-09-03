# AI 投资助手规划（Assistant）

> 目标：在 web 界面提供一个**常驻 AI 对话入口**，继承现有 LLM 能力，深度理解
> 「LLM 策略挖掘」全流程——能解释生成代码的意图、通过对话直接修改代码、
> 解读回测过程与结果，并回答量化研究相关问题。

## 一、现状盘点（可复用的积木）

| 能力 | 现状 | 位置 |
|------|------|------|
| LLM 多轮对话 | `chat_messages(system, msgs)` 已支持多轮 | `quantmind/ai/provider.py` |
| 代码生成+自修复 | `/strategy/draft/start` 后台任务 + 轮询，支持 history 修改轮 | `api/app.py`、draft 任务 |
| 沙箱校验 | AST 白名单 `validate_code` | `quantmind/ai/sandbox.py` |
| 代码手动编辑 | 页面24已有编辑器 + 「检验修改」+「恢复上一版」 | `web/pages/24_LLM策略挖掘.py` |
| 回测结果结构化数据 | per_symbol report / equity_curve / trade_list / gate / optim_detail | history JSON |
| 服务端状态持久化 | `/strategy/draft/state` GET/PUT/DELETE | api/app.py |
| **缺失** | 通用对话端点、上下文注入、对话持久化、代码意图解释、结果自动解读 | — |

**关键约束**：`LLMProvider` 只有 `chat/chat_messages`，无原生 function-calling，
需用 **JSON 动作协议**模拟工具调用；LLM 调用耗时 10~60s，必须复用
**后台任务 + task_id 轮询**模式（切页不中断）。

## 二、总体架构

```
Streamlit 前端                        FastAPI 后端
┌─────────────────────┐   POST /assistant/chat/start      ┌──────────────────┐
│ 💬 对话面板(页面24    │ ────────────────────────────────▶ │ AssistantService │
│   内嵌 + 独立页二期)  │   GET  /assistant/chat/status/... │  · 会话管理        │
│                     │                                   │  · 上下文组装      │
│  自动携带上下文：     │   上下文 = 系统提示 + 当前思想/     │  · JSON 动作解析   │
│  · val_idea         │   代码 + 最新回测摘要 + 对话历史     │  · 工具执行        │
│  · 当前代码          │                                   └──────┬───────────┘
│  · 最新回测结果摘要   │                                    ▼
└─────────────────────┘                     LLMProvider.chat_messages
                                                     │
                                    JSON: { reply, actions[] }
                                                     ▼
                              ┌────────────────────────────────────────┐
                              │ 工具（后端执行，结果喂回下一轮）             │
                              │  explain_code   → 代码意图解读            │
                              │  modify_code    → 复用 draft 流(+沙箱)   │
                              │  validate_code  → sandbox.check          │
                              │  run_backtest   → 复用 /strategy/validate │
                              │  get_result     → 读历史 run 结果         │
                              └────────────────────────────────────────┘
```

### 动作协议（弥补无 function-calling）

LLM �轮回复 JSON：

```json
{
  "reply": "给用户看的自然语言回复",
  "actions": [
    {"type": "modify_code", "instruction": "止损改成2%固定止损"},
    {"type": "validate_code", "code": "..."}
  ]
}
```

后端解析执行 actions → 把执行结果（成功/失败/回测摘要）作为新一轮 user 消息
喂回 LLM，最多 3 轮循环，最终把 `reply` + 动作结果一并返回前端。

## 三、三个核心场景设计

### 场景1：代码意图解释（生成后自动 + 随时可问）

- **触发**：draft 代码生成成功（sandbox_ok）后，后台自动发起一次"解释"任务
  （不阻塞主流程）；解释结果存入 draft state（按 code hash 缓存，同代码不重复调用）。
- **内容**：让 LLM 输出结构化解读——①策略逻辑概述 ②入场/离场/止损规则逐条
  对应代码位置 ③参数及默认值 ④与用户思想的偏差点（如有）⑤潜在风险
  （前视、未处理停牌、单边只有多头等）。
- **展示**：代码区上方新增「🧠 代码解读」折叠面板；对话面板可继续追问。

### 场景2：对话式修改代码

- 用户在对话面板说"止损改成2%，再加一个ATR跟踪止损"。
- Assistant 调用 `modify_code`：走现有 `/strategy/draft/start` 流
  （history = [assistant: 当前代码, user: 指令]），天然继承自修复轮次。
- 完成后：自动沙箱校验 → 更新代码区 → 用 `difflib.unified_diff` 生成
  **改动高亮**展示在对话气泡中（"改了 3 处：…"），保留「恢复上一版」能力。
- 校验失败时，assistant 自动带错误信息再修一轮（复用现有 repair 机制）。

### 场景3：回测结果解读

- **自动**：回测任务 success 后，后台生成一段「📊 AI 解读」：
  - 结果概览（达标情况、最/最优品种）；
  - **可靠性提示**（直接复用已有判定逻辑的口径）：样本量 <60 交易日/<20 笔、
    前3大盈利日集中度、OOS/IS 保持度、DSR<0.9、成本占比；
  - 改进建议（参数、周期、品种、逻辑层面各给 1 条）。
- **交互**：结果页四个 Tab 各有「问 AI」入口；用户可追问
  "IC0 年化 63% 而 IF0 是 -19%，正常吗？"——上下文已含摘要，
  assistant 能对照样本警示回答（正是此前用户遇到的真实疑问）。
- **上下文摘要模板**（控制 token）：概览表 KPI 行 + 门槛/优化判定 +
  集中度 + 样本天数/笔数；trade_list 只给统计（笔数、盈亏分布、均持仓时长）。

## 四、API 设计

```
POST /assistant/chat/start        # {message, session_id?, context_keys} → task_id
GET  /assistant/chat/status/{id}  # 轮询（复用现有任务框架）
GET  /assistant/sessions          # 会话列表
GET  /assistant/sessions/{id}     # 会话历史（服务端持久化，data_cache/assistant_sessions/）
DELETE /assistant/sessions/{id}
POST /assistant/explain_code      # {code} → 结构化解读（带 hash 缓存）
POST /assistant/explain_result    # {run_id} → 回测解读（同样缓存）
```

- 上下文由**前端只传 key**（如 `context: {code: true, last_result: run_id}`），
  后端自行读取 draft state / history 组装，避免大 payload 与前端伪造。
- 会话历史按用户+页面分 session；条数上限 50 轮，超限截断最早轮次。

## 五、前端设计

### P0：页面 24 内嵌对话面板（推荐先做）

- 代码区与结果区之间增加 `💬 AI 助手` expander：
  `st.chat_message` 渲染历史 + `st.chat_input` 输入；
  预设快捷按钮：「解释这段代码」「这个结果靠谱吗」「如何改进」「帮我改止损为2%」。
- 发消息 → `chat/start` → 复用 `@fragment(run_every=3)` 轮询模式。
- 对话中触发的代码修改/回测，直接复用页面现有的 draft/val 轮询与状态字段，
  **单一数据源仍是 val_generated_code / val_result**。

### P1/P2：独立「AI 助手」页

- 全局入口，可跨页引用上下文（仪表盘、风控、知识库）；
- 支持 assistant 主动给出**跳转链接**（"已为你打开策略挖掘页并填入思想"）。

## 六、分期实施

| 阶段 | 内容 | 预估 |
|------|------|------|
| **P0** | `/assistant/chat/start+status` + 动作协议（explain/modify/validate）+ 页面24对话面板 + 上下文注入 + 代码解释 | 1.5~2 天 |
| **P1** | 回测完成自动生成解读 + 代码改动 diff 高亮 + 会话持久化 + 快捷问题 + explain 缓存 | 1 天 |
| **P2** | 独立助手页 + run_backtest/get_result 工具 + 跨页上下文 + 流式(SSE)输出 | 2~3 天 |

## 七、风险与注意事项

1. **上下文膨胀**：回测结果/代码全文一次性塞入会撑爆 token——只注摘要，
   trade_list 只注统计；对话超 50 轮滚动截断。
2. **安全**：assistant 产出的任何代码**必须**过 `validate_code` 沙箱才能进入
   回测/入库路径，与手动编辑同等待遇，无旁路。
3. **成本控制**：解释/解读均按内容 hash 缓存；MockProvider 兜底（未配 LLM 时
   给出规则化的静态解读）。
4. **任务并发**：chat 任务与 draft/val 任务共用后台任务管理器，注意
   `_prune_tasks` 与任务 ID 命名空间隔离（`asst_` 前缀）。
5. **Streamlit 陷阱**（历史教训）：解读文本渲染用普通 if/else，避免三元表达式
   语句把 DeltaGenerator 文档打到页面；对话历史 key 与页面 23 缓存问题类似，
   错误轮次不做永久缓存。
