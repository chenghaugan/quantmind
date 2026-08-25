# 日期范围选择功能 - 实现总结

## 概述
为量化投研平台的核心功能模块添加了日期范围选择器，允许用户指定因子研究、策略回测、因子挖掘和端到端流水线的数据时间区间。

## 修改的文件

### 后端 Schema 层 (2 files)

#### 1. `quantmind/api/schemas.py`
- **FactorRequest**: 添加 `start: Optional[str] = None` 和 `end: Optional[str] = None` 字段
- **BacktestRequest**: 添加 `start: Optional[str] = None` 和 `end: Optional[str] = None` 字段
- 日期格式：ISO 8601 (YYYY-MM-DD)

#### 2. `quantmind/api/services/factor_service.py`
- **evaluate() 方法**: 
  - 更新缓存键以包含日期范围
  - 构造 HistoryRequest 时，如果提供了 start/end，则转换为 datetime 对象并传入
  - 使用 `**history_kwargs` 动态传参

#### 3. `quantmind/api/services/backtest_service.py`
- **run_backtest() 方法**:
  - 构造 HistoryRequest 时，如果提供了 start/end，则转换为 datetime 对象并传入
  - 使用 `**history_kwargs` 动态传参

### 前端 API 客户端层 (1 file)

#### 4. `quantmind/web/utils/api_client.py`
- **factor() 方法**: 添加 `start` 和 `end` 参数，传入后端请求
- **backtest() 方法**: 添加 `start` 和 `end` 参数，传入后端请求

### 前端页面层 (4 files)

#### 5. `quantmind/web/pages/3_因子研究.py`
- 在表单中添加日期范围选择器（开始日期、结束日期）
- 位置：左侧列，交易所选择下方
- 更新 evaluate() 函数签名，接收 start/end 参数
- 调用时将 date 对象转换为 ISO 格式字符串

#### 6. `quantmind/web/pages/4_策略回测.py`
- 在回测设置表单中添加日期范围选择器
- 位置：策略参数区块上方
- 标签：**回测区间**（可选，留空使用全部可用数据）
- 调用 APIClient.backtest() 时传入 start/end 参数

#### 7. `quantmind/quantmind/web/pages/18_因子挖掘流水线.py`
- 在右侧列添加日期范围选择器
- 位置：验证期占比下方，运行按钮上方
- 标签：**数据区间**（可选，留空使用全部数据）
- 使用唯一 key (`pipe_start`, `pipe_end`) 避免冲突
- 更新 payload 字典，包含 start/end 字段

#### 8. `quantmind/web/pages/20_端到端流水线.py`
- 在右侧列添加日期范围选择器
- 位置：验证期占比下方，代码校验阈值上方
- 标签：**数据区间**（可选，留空使用全部数据）
- 使用唯一 key (`e2e_start`, `e2e_end`) 避免冲突
- 更新 payload 字典，将硬编码的 `None` 替换为用户选择的日期

## UI 设计

### 布局模式
```python
st.markdown("**数据区间**（可选，留空使用全部数据）")
dc1, dc2 = st.columns(2)
with dc1:
    start_date = st.date_input("开始日期", value=None, format="YYYY-MM-DD")
with dc2:
    end_date = st.date_input("结束日期", value=None, format="YYYY-MM-DD")
```

### 设计原则
1. **可选性**: value=None 表示默认不限制，使用全部可用数据
2. **清晰提示**: 明确标注"可选，留空使用全部数据"
3. **一致性**: 所有页面使用相同的 UI 模式和标签文案
4. **唯一性**: 每个页面使用唯一的 key 避免 session_state 冲突

## 数据流

```
用户选择日期 → date_input 返回 date 对象
    ↓
date.isoformat() → "YYYY-MM-DD" 字符串
    ↓
API 客户端 → JSON payload (start/end 字段)
    ↓
后端 Schema → FactorRequest/BacktestRequest 接收字符串
    ↓
Service 层 → datetime.fromisoformat() 转换
    ↓
HistoryRequest → 传入 DataManager 获取数据
```

## 向后兼容性

- 所有日期参数都是 Optional，默认为 None
- None 表示不限制日期范围，使用全部可用数据
- 现有代码和 API 调用无需修改即可继续工作
- 缓存键包含日期范围，确保不同日期范围的结果正确缓存

## 测试建议

1. **因子研究页面**:
   - 不选日期 → 使用全部数据
   - 选择日期范围 → 仅使用指定区间数据
   - 验证 IC/IR 计算结果是否因数据范围变化

2. **策略回测页面**:
   - 不选日期 → 使用全部数据回测
   - 选择日期范围 → 仅在指定区间回测
   - 验证净值曲线和绩效指标

3. **因子挖掘流水线**:
   - 验证 payload 中 start/end 正确传递
   - 检查后端日志确认日期参数生效

4. **端到端流水线**:
   - 验证异步任务中日期参数正确传递
   - 检查因子挖掘和回测阶段是否使用指定日期

## 注意事项

1. **日期格式**: 统一使用 ISO 8601 (YYYY-MM-DD)
2. **时区处理**: 当前实现假设本地时区，如需支持时区需额外处理
3. **数据可用性**: 如果指定日期范围内无数据，后端会返回相应错误
4. **性能影响**:  larger 日期范围可能导致更长的计算时间

## 后续优化建议

1. 添加常用日期范围快捷选择（如"最近1年"、"最近3年"、"全部"）
2. 添加日期范围验证（开始日期 < 结束日期）
3. 显示所选日期范围内的数据点数量
4. 支持更精细的时间选择（如小时、分钟级别）
