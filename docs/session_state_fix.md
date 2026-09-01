# LLM策略挖掘页面 - Session State 持久化修复

## 问题描述

**用户反馈**：
> "web界面-llm策略挖掘，我从其他界面切回，怎么又变回了默认的界面，原来跑过的一次策略思想没有保存"

**症状**：
1. 用户输入自定义策略思想
2. 生成策略代码
3. 切换到其他页面（如"仪表盘"、"数据管理"等）
4. 切回"LLM策略挖掘"页面
5. **发现**：
   - 策略思想输入框显示默认模板，而非用户输入
   - 生成的代码丢失
   - 回测参数选择丢失
   - 视图选择（运行/历史）重置

## 第一性原理分析

### Streamlit Session State 工作机制

Streamlit 的多页面应用中，session_state 的生命周期：

```
浏览器会话开始
    ↓
创建 Session State（跨页面共享）
    ↓
页面 A 渲染 → 读取/写入 session_state
    ↓
切换到页面 B → 页面 A 的 widget 状态丢失
    ↓
页面 B 渲染 → 读取/写入 session_state
    ↓
切换回页面 A → 页面 A 重新渲染
    ↓
Widget 需要从 session_state 恢复状态
```

### 关键问题点

#### 问题 1：Widget 状态未绑定到 Session State

**原代码**：
```python
_view = st.radio("视图", ["🚀 运行策略挖掘", "📜 历史运行报告"], horizontal=True)
```

**问题**：
- 没有 `key` 参数
- Radio 按钮的状态不会自动保存到 session_state
- 页面重新渲染时，状态丢失，默认选择第一项

**修复**：
```python
if "val_view" not in st.session_state:
    st.session_state.val_view = "🚀 运行策略挖掘"
_view = st.radio(
    "视图",
    ["🚀 运行策略挖掘", "📜 历史运行报告"],
    horizontal=True,
    key="val_view",  # ← 关键：绑定到 session_state
)
```

#### 问题 2：Text Area 状态恢复不可靠

**原代码**：
```python
idea = st.text_area(
    "💡 策略思想（自然语言规则，LLM 将编程实现）",
    height=120,
    placeholder="描述完整的交易规则：入场、离场、止损、参数…",
    key="val_idea",
)
```

**潜在问题**：
- 虽然使用了 `key`，但在某些 Streamlit 版本中，widget 可能不会正确恢复 session_state 的值
- 用户输入后切换页面，再切回时可能显示默认值而非用户输入

**修复**：
- 确保在 widget 渲染前显式初始化 session_state
- 添加状态恢复提示，让用户知道会话已恢复

#### 问题 3：缺乏状态恢复反馈

**问题**：
- 用户切换页面后，不知道之前的状态是否保留
- 如果状态丢失，用户无法察觉，直到发现输入框显示默认值

**修复**：
- 检测是否有未完成的会话（`draft_pending` 或 `val_generated_code`）
- 显示蓝色提示框："🔄 检测到上次未完成的会话，已自动恢复"

## 修复方案

### 1. Radio 按钮持久化

**文件**：`quantmind/web/pages/24_LLM策略挖掘.py`

**修改位置**：第 309 行附近

```python
# 视图切换（持久化到 session_state）
if "val_view" not in st.session_state:
    st.session_state.val_view = "🚀 运行策略挖掘"
_view = st.radio(
    "视图",
    ["🚀 运行策略挖掘", "📜 历史运行报告"],
    horizontal=True,
    key="val_view",
)
```

### 2. 状态恢复提示

**修改位置**：第 445 行附近

```python
# 检查是否有未完成的会话
_has_pending_draft = bool(st.session_state.get("draft_pending"))
_has_generated_code = st.session_state.val_generated_code is not None

if _has_pending_draft or _has_generated_code:
    st.info(
        "🔄 检测到上次未完成的会话，已自动恢复。"
        + ("继续审阅生成的代码，或点击"清空重新开始"。" if _has_pending_draft else "")
    )
```

### 3. 确保所有关键 Widget 都有 Key

**已持久化的 Widget**：
- `val_view` - 视图选择
- `val_idea` - 策略思想输入
- `val_symbols` - 测试品种选择
- `val_interval` - 数据周期选择
- `val_start` / `val_end` - 起止日期
- `use_template_toggle` - 是否使用预置模板
- `template_select` - 预置模板选择
- `draft_feedback` - 修改意见输入

## 测试验证

### 测试脚本

已创建测试脚本：`test_session_persistence.py`

**运行方式**：
```bash
cd /home/dylan/docker/quantmind
streamlit run test_session_persistence.py
```

### 手动测试步骤

1. **打开 LLM策略挖掘页面**
   ```
   http://localhost:8502/LLM策略挖掘
   ```

2. **输入自定义策略思想**（不要使用默认模板）
   ```
   示例：双均线策略，5日均线上穿20日均线时买入，下穿时卖出，止损5%
   ```

3. **点击"生成策略代码"**
   - 等待代码生成完成（约 10-60 秒）

4. **切换到其他页面**
   - 点击左侧导航栏的"仪表盘"或"数据管理"

5. **切回 LLM策略挖掘页面**

6. **验证以下内容**：
   - ✓ 页面顶部显示蓝色提示："🔄 检测到上次未完成的会话，已自动恢复"
   - ✓ 策略思想输入框显示你刚才输入的内容（不是默认模板）
   - ✓ 代码审阅区域显示生成的代码
   - ✓ 视图选择保持在"🚀 运行策略挖掘"

7. **继续测试阶段二**：
   - 点击"✅ 确认代码，进入阶段二"
   - 选择测试品种（如 IC0, IF0）
   - 选择数据周期（如 1h）
   - 切换到其他页面
   - 切回，验证参数是否保留

### 预期结果

**成功**：
- 所有用户输入和选择都保留
- 页面显示状态恢复提示
- 可以继续之前的工作

**失败**：
- 策略思想显示默认模板
- 生成的代码丢失
- 需要重新开始

## 技术细节

### Session State 生命周期

```
用户输入 → Widget 更新 session_state
    ↓
页面重新渲染 → Widget 从 session_state 读取值
    ↓
切换到其他页面 → 当前页面的 widget 销毁
    ↓
切回当前页面 → 页面重新渲染
    ↓
Widget 重新创建 → 从 session_state 恢复值
```

### 关键点

1. **Widget 必须有 `key` 参数**
   - 没有 `key` 的 widget 不会自动同步到 session_state
   - 页面重新渲染时状态丢失

2. **显式初始化 session_state**
   - 在 widget 渲染前检查并初始化 session_state
   - 避免 KeyError

3. **提供状态恢复反馈**
   - 让用户知道会话已恢复
   - 增强用户体验

## 后续优化建议

### 1. 添加"保存草稿"功能

允许用户手动保存当前的策略思想和代码，即使关闭浏览器也能恢复。

```python
if st.button("💾 保存草稿"):
    # 保存到本地文件或数据库
    save_draft(st.session_state.val_idea, st.session_state.val_generated_code)
    st.success("草稿已保存")
```

### 2. 添加"历史草稿"列表

显示用户之前保存的草稿，可以选择恢复。

```python
drafts = load_drafts()
selected_draft = st.selectbox("选择历史草稿", drafts)
if st.button("恢复草稿"):
    restore_draft(selected_draft)
```

### 3. 自动保存

使用 `st.fragment(run_every=60)` 每 60 秒自动保存一次草稿。

```python
@st.fragment(run_every=60)
def auto_save():
    save_draft(st.session_state.val_idea, st.session_state.val_generated_code)
```

## 相关文件

- `quantmind/web/pages/24_LLM策略挖掘.py` - 主页面文件
- `test_session_persistence.py` - 测试脚本
- `docs/session_state_fix.md` - 本文档

## 参考资料

- [Streamlit Session State 文档](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state)
- [Streamlit 多页面应用](https://docs.streamlit.io/develop/concepts/multipage-apps)
- [Streamlit Widget 状态管理](https://docs.streamlit.io/develop/concepts/architecture/session-state#widget-state)

## 版本信息

- **修复日期**：2026-08-31
- **修复版本**：v1.0
- **Streamlit 版本**：1.62.0
- **Python 版本**：3.13

## 联系方式

如有问题，请联系开发团队。
