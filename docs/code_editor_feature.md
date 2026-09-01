# 策略代码编辑器功能

## 功能概述

LLM策略挖掘页面现在支持**直接编辑生成的策略代码**，并提供实时沙箱检验功能。

## 使用流程

### 1. 生成策略代码

1. 在"策略思想"输入框中描述你的策略逻辑
2. 点击"🧬 生成策略代码"
3. 等待LLM生成代码（约10-60秒）

### 2. 编辑代码

代码生成后，页面会显示一个**可编辑的代码编辑器**：

```
✅ 策略代码已生成

📝 代码编辑器（可直接修改，修改后点击"重新检验"）
┌─────────────────────────────────────────┐
│ from quantmind.strategy.base import ... │
│                                         │
│ class MyStrategy(CtaTemplate):          │
│     def on_bar(self, bar):              │
│         # 你的策略逻辑                   │
│         ...                             │
└─────────────────────────────────────────┘

[🔍 重新检验]  [🗑️ 清空重新开始]
```

### 3. 重新检验

编辑代码后，点击**"🔍 重新检验"**按钮：

- ✅ **检验通过**：代码已更新，可以继续进入阶段二
- ❌ **检验失败**：显示错误信息，需要修复后重新检验

### 4. 进入阶段二

检验通过后，点击**"✅ 确认代码，进入阶段二"**，选择回测参数并开始回测。

## 沙箱检验规则

沙箱检验会检查以下内容：

### 1. 语法检查
- Python语法是否正确
- 是否有明显的语法错误

### 2. 安全检查
- 禁止导入危险模块（如 `os`, `sys`, `subprocess` 等）
- 禁止使用危险函数（如 `eval`, `exec`, `open` 等）
- 禁止访问危险属性（如 `__globals__`, `__builtins__` 等）

### 3. 结构检查
- 必须继承 `CtaTemplate` 基类
- 必须实现 `on_bar` 方法
- 类名必须以 `Strategy` 结尾

### 4. 白名单导入
只允许导入以下模块：
- `quantmind.strategy.base`
- `quantmind.core.utility`
- `numpy`
- `pandas`
- `math`

## 常见错误及修复

### 错误 1：未继承 CtaTemplate

```python
# ❌ 错误
class MyStrategy:
    def on_bar(self, bar):
        pass

# ✅ 正确
from quantmind.strategy.base import CtaTemplate

class MyStrategy(CtaTemplate):
    def on_bar(self, bar):
        pass
```

### 错误 2：导入危险模块

```python
# ❌ 错误
import os
import subprocess

# ✅ 正确
# 只使用白名单内的模块
```

### 错误 3：使用危险函数

```python
# ❌ 错误
eval("1 + 1")
exec("print('hello')")

# ✅ 正确
# 使用安全的Python语法
result = 1 + 1
```

## 编辑建议

### 1. 修改参数

```python
class MyStrategy(CtaTemplate):
    parameters = ['fast_period', 'slow_period', 'stop_loss']
    
    def __init__(self, context, setting=None):
        self.fast_period = 10  # 可以修改默认值
        self.slow_period = 30
        self.stop_loss = 0.02
        super().__init__(context, setting)
```

### 2. 调整逻辑

```python
def on_bar(self, bar):
    # 修改入场条件
    if self.fast_ma > self.slow_ma * 1.01:  # 添加1%的过滤
        self.buy(bar.symbol, bar.exchange, bar.close, 1)
```

### 3. 添加风控

```python
def on_bar(self, bar):
    # 添加止损逻辑
    if self.position and self.position.pnl < -self.stop_loss * self.position.cost:
        self.sell(bar.symbol, bar.exchange, bar.close, self.position.volume)
        return
    
    # 原有逻辑
    ...
```

## 技术实现

### 后端API

```python
@app.post("/strategy/draft/validate")
async def strategy_draft_validate(payload: Dict[str, Any]):
    """沙箱校验策略代码"""
    code = payload.get("code", "")
    ok, err, errors = compile_strategy(code, require_base="CtaTemplate")
    return {"ok": ok, "error": err, "errors": errors}
```

### 前端交互

1. 用户编辑代码 → `st.text_area` 捕获修改
2. 点击"重新检验" → 调用 `/strategy/draft/validate` API
3. 检验通过 → 更新 `session_state.val_generated_code`
4. 页面重新渲染 → 显示最新代码

## 注意事项

1. **编辑后必须重新检验**：修改代码后，必须点击"重新检验"才能生效
2. **检验失败不影响原代码**：检验失败时，原代码仍然保留，可以继续修改
3. **清空重新开始**：点击"清空重新开始"会清除所有状态，包括编辑的代码
4. **代码持久化**：编辑的代码会自动保存到 session_state，切换页面不会丢失

## 示例：修改双均线策略

### 原始代码（LLM生成）

```python
from quantmind.strategy.base import CtaTemplate
from quantmind.core.utility import ArrayManager

class DualMaStrategy(CtaTemplate):
    parameters = ['fast', 'slow']
    
    def __init__(self, context, setting=None):
        self.fast = 5
        self.slow = 20
        self.am_fast = None
        self.am_slow = None
        super().__init__(context, setting)
    
    def on_bar(self, bar):
        if self.am_fast is None:
            self.am_fast = ArrayManager(self.fast)
            self.am_slow = ArrayManager(self.slow)
        
        self.am_fast.update(bar.close)
        self.am_slow.update(bar.close)
        
        if not (self.am_fast.inited and self.am_slow.inited):
            return
        
        fast_ma = self.am_fast.mean()
        slow_ma = self.am_slow.mean()
        
        if fast_ma > slow_ma:
            self.set_target(bar.symbol, 1)
        elif fast_ma < slow_ma:
            self.set_target(bar.symbol, -1)
```

### 编辑后（添加止损和止盈）

```python
from quantmind.strategy.base import CtaTemplate
from quantmind.core.utility import ArrayManager

class DualMaStrategy(CtaTemplate):
    parameters = ['fast', 'slow', 'stop_loss', 'take_profit']
    
    def __init__(self, context, setting=None):
        self.fast = 5
        self.slow = 20
        self.stop_loss = 0.02  # 2%止损
        self.take_profit = 0.05  # 5%止盈
        self.am_fast = None
        self.am_slow = None
        self.entry_price = None
        super().__init__(context, setting)
    
    def on_bar(self, bar):
        if self.am_fast is None:
            self.am_fast = ArrayManager(self.fast)
            self.am_slow = ArrayManager(self.slow)
        
        self.am_fast.update(bar.close)
        self.am_slow.update(bar.close)
        
        if not (self.am_fast.inited and self.am_slow.inited):
            return
        
        # 检查止损止盈
        if self.entry_price:
            pnl = (bar.close - self.entry_price) / self.entry_price
            
            if pnl <= -self.stop_loss:
                self.set_target(bar.symbol, 0)
                self.entry_price = None
                return
            
            if pnl >= self.take_profit:
                self.set_target(bar.symbol, 0)
                self.entry_price = None
                return
        
        fast_ma = self.am_fast.mean()
        slow_ma = self.am_slow.mean()
        
        # 金叉做多
        if fast_ma > slow_ma and self.pos == 0:
            self.set_target(bar.symbol, 1)
            self.entry_price = bar.close
        
        # 死叉平仓
        elif fast_ma < slow_ma and self.pos > 0:
            self.set_target(bar.symbol, 0)
            self.entry_price = None
```

## 相关文件

- 前端页面：`quantmind/web/pages/24_LLM策略挖掘.py`
- 后端API：`quantmind/api/app.py` (第 1290-1310 行)
- 沙箱检验：`quantmind/ai/sandbox.py`

## 版本历史

- **2026-08-31**：添加代码编辑和重新检验功能
