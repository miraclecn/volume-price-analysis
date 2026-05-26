# A股多层级量价结构识别系统 Spec

## 1. 项目名称

英文名称：

```text
A-Share Multi-Level Volume Price Structure Recognizer
```

中文名称：

```text
A股多层级量价结构识别系统
```

模块建议名称：

```text
vpa_structure_recognizer
```

如果集成到 `alpha-finder-v2` 中，建议作为独立研究模块：

```text
modules/research/vpa_structure_recognizer/
```

## 2. 核心目标

构建一个基于日线 K 线数据的量价结构识别系统。

系统目标不是直接预测涨跌，也不是根据单根 K 线给出买卖信号，而是：

```text
1. 从全A市场、板块、个股三个层级自上而下分析量价结构。
2. 对每一个交易日，在不同观察窗口下分别生成量价标签。
3. 对 10 / 20 / 30 / 60 / 120 / 240 日等多窗口量价标签进行序列分析。
4. 判断当前市场、板块、个股分别处于什么趋势背景和量价结构阶段。
5. 输出候选股票、风险股票、板块强弱、个股结构评级和后续验证条件。
6. 最终导出 Excel，用于人工复盘和后效验证。
```

## 3. 设计原则

### 3.1 价格波动必须百分比化

所有价格振动、实体、上下影线、涨跌幅都必须使用百分比，而不是实际价格差。

禁止直接使用：

```text
high - low
close - open
```

必须使用：

```text
ret_pct = close / prev_close - 1
range_pct = (high - low) / prev_close
body_pct = abs(close - open) / prev_close
upper_shadow_pct = (high - max(open, close)) / prev_close
lower_shadow_pct = (min(open, close) - low) / prev_close
```

同时保留 K 线内部结构比例：

```text
body_ratio = abs(close - open) / (high - low)
upper_shadow_ratio = (high - max(open, close)) / (high - low)
lower_shadow_ratio = (min(open, close) - low) / (high - low)
close_position = (close - low) / (high - low)
```

其中：

```text
close_position > 0.7：收盘靠近高位，买方占优
close_position < 0.3：收盘靠近低位，卖方占优
```

### 3.2 成交量放量/缩量必须按窗口动态计算

成交量不能只用固定 20 日均量评价。不同窗口必须使用各自窗口的平均成交量。

例如：

```text
vol_rvol_10 = volume / avg(volume, 10)
vol_rvol_20 = volume / avg(volume, 20)
vol_rvol_30 = volume / avg(volume, 30)
vol_rvol_60 = volume / avg(volume, 60)
vol_rvol_120 = volume / avg(volume, 120)
vol_rvol_240 = volume / avg(volume, 240)
```

同一交易日，在不同窗口下可能得到完全不同的量价标签。

例如：

```text
相对10日均量：明显放量
相对20日均量：温和放量
相对60日均量：正常量
相对120日均量：缩量
```

因此：

```text
量价标签必须带 window_n 字段。
```

### 3.3 每个成交日必须生成多窗口标签

同一只股票、同一个交易日，必须分别计算不同窗口下的量价标签。

示例：

```text
date        code      window_n    label
2026-05-20  600000    10          明显放量上攻
2026-05-20  600000    20          温和放量上涨
2026-05-20  600000    60          正常量反弹
2026-05-20  600000    120         长期缩量反弹
```

单日标签不是全局标签，而是：

```text
当前交易日在当前窗口背景下的量价状态。
```

### 3.4 单日标签只能描述量价事实，不能直接定义吸筹/派发

单日标签只能描述现象。

允许的标签类型：

```text
放量上涨确认
放量下跌确认
放量但上涨效率低
放量但下跌效率低
缩量上涨
缩量下跌
高量低波动
低量高波动
长上影供应增强
长下影承接增强
突破后回落
跌破后收回
窄幅缩量
正常上涨
正常下跌
```

禁止单日标签直接使用：

```text
吸筹
派发
出货
洗盘
主力建仓
主力护盘
主力拉升
```

原因：吸筹、派发、拉升、破位等属于多日结构判断，不能由单根 K 线直接给出。

### 3.5 趋势判断必须使用更大窗口

分析某个窗口的量价结构时，必须用更大窗口判断趋势背景。

建议映射：

```text
10日结构  -> 使用30日或60日判断背景
20日结构  -> 使用60日判断背景
30日结构  -> 使用120日判断背景
60日结构  -> 使用120日或240日判断背景
120日结构 -> 使用240日判断背景
```

例如：分析20日量价结构时，不能只看20日内部涨跌，必须判断这20日处于60日趋势的什么位置。

### 3.6 异常标签表达多空强弱，但不直接等于未来走势

每个异常标签可以给出一个当前窗口下的多空强弱倾向。

例如：

```text
长下影承接增强 -> bull_bear_score = +1
放量长上影供应增强 -> bull_bear_score = -1
放量上涨确认 -> bull_bear_score = +1
放量下跌确认 -> bull_bear_score = -1
高量低波动 -> bull_bear_score = 0 或根据位置再解释
```

但这个分数只代表当前窗口下该交易日多空力量的即时表现，不能直接推导未来走势。

必须通过：

```text
多日标签序列
趋势背景
板块环境
全A环境
历史后效验证
```

才能判断它是否具有预测价值。

### 3.7 系统必须自上而下

分析顺序必须是：

```text
全A市场 -> 板块 -> 个股
```

不能直接只看个股。个股如果违背全A或板块趋势，应降低评级。

例如：

```text
全A弱势 + 板块弱势 + 个股强势
```

应标记为：

```text
逆势个股，降低评级，仅观察。
```

如果：

```text
全A强势 + 板块强势 + 个股强势
```

应标记为：

```text
市场、板块、个股三层共振，评级提高。
```

### 3.8 输出顺序也必须自上而下

分析报告必须按照：

```text
1. 全A市场结构
2. 板块结构
3. 个股结构
4. 综合评级
5. 后续确认条件
```

不能直接从个股开始。筛选股票时同样必须先判断全A环境，再筛选趋势良好的板块，最后从强板块中筛选结构良好的个股。

## 4. 系统输入

### 4.1 个股日线数据

至少需要字段：

```text
date
code
open
high
low
close
prev_close
volume
amount
turnover_rate
is_st
is_paused
limit_up
limit_down
industry_code
industry_name
```

价格建议使用前复权价格。成交量使用原始成交量。

### 4.2 板块数据

优先使用已有板块指数 K 线。如果没有板块指数，则通过板块内成分股聚合生成。

字段：

```text
date
sector_code
sector_name
open
high
low
close
prev_close
volume
amount
advancers_count
decliners_count
limit_up_count
limit_down_count
member_count
```

板块 K 线可以使用两种方式：

```text
1. 板块指数K线
2. 板块内个股等权收益聚合K线
```

第一版优先使用板块指数。如果没有板块指数，则使用等权聚合。

### 4.3 全A市场数据

全A层面不应只看上证指数。建议构建全A市场宽度数据：

```text
date
all_a_equal_weight_open
all_a_equal_weight_high
all_a_equal_weight_low
all_a_equal_weight_close
total_amount
total_volume
advancers_count
decliners_count
limit_up_count
limit_down_count
new_high_count_20
new_low_count_20
new_high_count_60
new_low_count_60
strong_stock_ratio
weak_stock_ratio
median_ret_pct
```

全A也需要生成自己的多窗口量价标签和结构状态。

## 5. 窗口设计

默认窗口：

```text
10
20
30
60
120
240
```

窗口用途：

```text
10日：短线变化
20日：主行为窗口
30日：结构确认窗口
60日：中期背景
120日：大级别趋势
240日：长期位置
```

父窗口映射：

```text
10  -> 30 或 60
20  -> 60
30  -> 120
60  -> 240
120 -> 240
240 -> 240
```

## 6. 数据表设计

### 6.1 基础特征表

表名：

```text
vpa_features
```

字段：

```text
date
scope_type        -- market / sector / stock
scope_id          -- ALL_A / sector_code / stock_code
window_n
open
high
low
close
prev_close
volume
amount
ret_pct
range_pct
body_pct
upper_shadow_pct
lower_shadow_pct
body_ratio
upper_shadow_ratio
lower_shadow_ratio
close_position
vol_ma_n
vol_rvol_n
range_pct_ma_n
range_rvol_n
body_pct_ma_n
body_rvol_n
price_high_n
price_low_n
price_position_n
ma_n
ma_slope_n
```

主键：

```text
date + scope_type + scope_id + window_n
```

### 6.2 趋势背景表

表名：

```text
vpa_trend_context
```

字段：

```text
date
scope_type
scope_id
window_n
parent_window_n
parent_high
parent_low
parent_price_position
parent_ma
parent_ma_slope
trend_label
position_label
trend_strength_score
```

趋势标签：

```text
UPTREND
DOWNTREND
SIDEWAYS
RECOVERING
WEAKENING
UNKNOWN
```

位置标签：

```text
LOW
MID_LOW
MID
MID_HIGH
HIGH
```

计算：

```text
parent_price_position = (close - parent_low) / (parent_high - parent_low)
```

分级：

```text
0.00 - 0.25 -> LOW
0.25 - 0.45 -> MID_LOW
0.45 - 0.65 -> MID
0.65 - 0.85 -> MID_HIGH
0.85 - 1.00 -> HIGH
```

### 6.3 多窗口单日量价标签表

表名：

```text
vpa_bar_context_labels
```

字段：

```text
date
scope_type
scope_id
window_n
parent_window_n
raw_label
normal_or_abnormal
volume_level
price_result_level
efficiency_level
bull_bear_score
supply_score
demand_score
volatility_score
description
```

volume_level：

```text
LOW_VOLUME
NORMAL_VOLUME
MILD_HIGH_VOLUME
HIGH_VOLUME
EXTREME_HIGH_VOLUME
```

成交量分级：

```text
vol_rvol_n < 0.7       -> LOW_VOLUME
0.7 - 1.2              -> NORMAL_VOLUME
1.2 - 1.8              -> MILD_HIGH_VOLUME
1.8 - 2.5              -> HIGH_VOLUME
> 2.5                  -> EXTREME_HIGH_VOLUME
```

### 6.4 多日序列结构表

表名：

```text
vpa_sequence_stats
```

字段：

```text
date
scope_type
scope_id
window_n
parent_window_n
normal_count
abnormal_count
abnormal_ratio
bullish_label_count
bearish_label_count
neutral_label_count
support_label_count
supply_label_count
high_volume_up_count
high_volume_down_count
high_volume_stall_count
long_upper_shadow_count
long_lower_shadow_count
low_volume_pullback_count
low_volume_rebound_count
breakout_like_count
breakdown_like_count
last_part_bull_score
previous_part_bull_score
bull_score_change
sequence_pattern
sequence_strength_score
```

说明：last_part 可以取最近 window_n / 2，previous_part 取前 window_n / 2。例如 20 日窗口：last_part = 最近10日，previous_part = 前10日。

### 6.5 阶段识别表

表名：

```text
vpa_structure_state
```

字段：

```text
date
scope_type
scope_id
state_10
state_20
state_30
state_60
state_120
state_240
final_state
trend_background
position_background
market_score
sector_score
self_score
relative_strength_score
resonance_score
final_rating
confidence
main_features
risk_flags
bullish_confirm_condition
bearish_invalidate_condition
```

final_state 可选：

```text
WEAK_DOWNTREND
DECLINE_EXHAUSTION
LOW_LEVEL_SUPPORT
POSSIBLE_ACCUMULATION
BREAKOUT_ATTEMPT
CONFIRMED_BREAKOUT
HEALTHY_UPTREND
ACCELERATING_UPTREND
HIGH_LEVEL_SUPPLY
POSSIBLE_DISTRIBUTION
BREAKDOWN
UNCLEAR
```

注意：`POSSIBLE_ACCUMULATION` / `POSSIBLE_DISTRIBUTION` 只能由多窗口序列和趋势位置共同判断，不能由单日标签生成。

## 7. 单日量价标签规则

### 7.1 量价正常

#### NORMAL_UP_CONFIRM

条件：

```text
ret_pct > 0
vol_rvol_n >= 1.0
body_ratio >= 0.45
close_position >= 0.65
```

含义：上涨有成交量支持，价格结果与成交量匹配。

#### NORMAL_DOWN_CONFIRM

条件：

```text
ret_pct < 0
vol_rvol_n >= 1.0
body_ratio >= 0.45
close_position <= 0.35
```

含义：下跌有成交量支持，价格结果与成交量匹配。

#### LOW_VOLUME_SMALL_MOVE

条件：

```text
vol_rvol_n < 0.8
range_rvol_n < 0.9
abs(ret_pct) 较小
```

含义：缩量窄幅，市场暂时安静。

### 7.2 量价异常

#### HIGH_VOLUME_LOW_PROGRESS

条件：

```text
vol_rvol_n >= 1.8
range_rvol_n <= 0.9
body_ratio <= 0.35
```

含义：高成交量没有产生足够价格推进，投入产出不匹配。

#### HIGH_VOLUME_UPPER_SUPPLY

条件：

```text
vol_rvol_n >= 1.5
upper_shadow_ratio >= 0.45
close_position <= 0.6
```

含义：上方供应增强，价格冲高后回落。

#### HIGH_VOLUME_LOWER_SUPPORT

条件：

```text
vol_rvol_n >= 1.5
lower_shadow_ratio >= 0.45
close_position >= 0.4
```

含义：下方承接增强，价格下探后被拉回。

#### LOW_VOLUME_BIG_UP

条件：

```text
vol_rvol_n < 0.8
ret_pct > 0
range_rvol_n >= 1.2
close_position >= 0.7
```

含义：缩量大涨，价格上涨缺乏当前窗口成交量确认。

#### LOW_VOLUME_BIG_DOWN

条件：

```text
vol_rvol_n < 0.8
ret_pct < 0
range_rvol_n >= 1.2
close_position <= 0.3
```

含义：缩量大跌，价格下跌缺乏当前窗口成交量确认。

#### BREAKOUT_PULLBACK

条件：

```text
盘中突破 window_n 内高点
但收盘回到高点下方
upper_shadow_ratio 较高
```

含义：突破失败或突破后供应增强。

#### BREAKDOWN_RECOVERY

条件：

```text
盘中跌破 window_n 内低点
但收盘回到低点上方
lower_shadow_ratio 较高
```

含义：跌破后被拉回，下方承接增强。

## 8. 多日序列模式识别

### 8.1 下跌衰竭序列

条件示例：

```text
parent_trend = DOWNTREND
parent_position in LOW / MID_LOW
最近 window_n 内：
- 放量下跌确认次数下降
- 高量下影承接次数增加
- 缩量下跌次数增加
- 价格创新低能力减弱
- 最近半窗口 bull_bear_score 高于前半窗口
```

输出：

```text
sequence_pattern = DECLINE_EXHAUSTION_PATTERN
```

解释：卖方仍有力量，但价格破坏力下降，低位承接开始增强。

### 8.2 低位承接增强序列

条件示例：

```text
parent_position in LOW / MID_LOW
support_label_count 较多
supply_label_count 较少
low_volume_pullback_count 增加
最近 window_n 内低点不再明显下移
```

输出：

```text
sequence_pattern = LOW_LEVEL_SUPPORT_PATTERN
```

解释：低位需求增强，卖压减弱，但尚未确认趋势启动。

### 8.3 健康上涨序列

条件示例：

```text
parent_trend = UPTREND or RECOVERING
上涨日量能不弱
回调日缩量
放量长阴较少
供应型异常较少
```

输出：

```text
sequence_pattern = HEALTHY_UPTREND_PATTERN
```

解释：上涨有量，回调无量，趋势相对健康。

### 8.4 高位供应增强序列

条件示例：

```text
parent_position in HIGH / MID_HIGH
supply_label_count 增加
high_volume_stall_count 增加
long_upper_shadow_count 增加
价格上涨效率下降
```

输出：

```text
sequence_pattern = HIGH_LEVEL_SUPPLY_PATTERN
```

解释：上方供应增强，价格推进效率下降。

### 8.5 疑似派发序列

条件示例：

```text
parent_position in HIGH / MID_HIGH
前期已有明显上涨
高量低波动频繁出现
长上影供应频繁出现
缩量反弹增加
跌破短期箱体或 MA20
```

输出：

```text
sequence_pattern = POSSIBLE_DISTRIBUTION_PATTERN
```

解释：高位买盘推进困难，供应逐渐占优。

### 8.6 假突破序列

条件示例：

```text
先出现突破类标签
随后 1-3 日跌回箱体
突破当日有高量长上影或高量低进展
后续反弹缩量
```

输出：

```text
sequence_pattern = FALSE_BREAKOUT_PATTERN
```

解释：突破没有得到后续需求确认，存在诱多或失败风险。

## 9. 自上而下分析流程

### 9.1 全A市场分析

第一步分析全A。

输出：

```text
market_final_state
market_trend_background
market_risk_appetite
market_main_features
```

全A评分：

```text
market_score = 
趋势得分
+ 市场宽度得分
+ 成交额状态得分
+ 强势股比例得分
- 跌停/弱势股风险扣分
```

市场状态：

```text
MARKET_STRONG
MARKET_NEUTRAL_POSITIVE
MARKET_SIDEWAYS
MARKET_WEAK
MARKET_DECLINE_EXHAUSTION
MARKET_HIGH_RISK
```

### 9.2 板块分析

只有在全A不是极弱时，才筛选板块。

板块评分：

```text
sector_score =
板块趋势得分
+ 板块量价序列得分
+ 相对全A强弱
+ 板块内部个股共振
```

板块状态：

```text
SECTOR_STRONG
SECTOR_IMPROVING
SECTOR_SIDEWAYS
SECTOR_WEAK
SECTOR_HIGH_SUPPLY
```

### 9.3 个股分析

个股评分：

```text
stock_score =
个股结构得分
+ 个股相对板块强弱
+ 个股多窗口共振
+ 个股风险扣分
```

如果个股违背全A或板块，降低评级。

规则：

```text
if market_score < 40:
    final_rating 降一级

if sector_score < 40:
    final_rating 降一级

if stock stronger than sector but sector weak:
    标记为逆势个股，仅观察

if market_score > 60 and sector_score > 60 and stock_score > 60:
    共振评分提高
```

## 10. 综合评级

最终评级：

```text
A：全A、板块、个股三层共振，结构良好
B：个股结构良好，板块或市场一般
C：结构不明确，仅观察
D：个股结构偏弱或环境不支持
E：高风险结构，避免主观看多
```

评级对应：

```text
final_score >= 80 -> A
65 - 79           -> B
50 - 64           -> C
35 - 49           -> D
< 35              -> E
```

final_score 建议：

```text
final_score =
market_score * 0.25
+ sector_score * 0.30
+ stock_score * 0.35
+ resonance_score * 0.10
```

如果系统用于风险识别，可以调整为更重视 market_score。

## 11. Excel 输出

最终导出一个 Excel 文件：

```text
vpa_structure_report_YYYYMMDD.xlsx
```

包含以下 sheet：

```text
1. 全A市场结构
2. 板块结构排名
3. 强势板块候选
4. 个股结构总表
5. 三层共振个股
6. 低位承接增强个股
7. 健康上涨个股
8. 高位供应风险个股
9. 放量破位风险个股
10. 个股多窗口标签明细
11. 后效验证数据
```

## 12. 报告输出格式

单只股票分析报告必须按以下结构：

```text
【1. 全A市场结构】
当前全A状态：
全A趋势背景：
全A量价特点：
市场风险偏好：
结论：

【2. 所属板块结构】
板块名称：
板块趋势：
板块量价结构：
板块相对全A强弱：
板块内部共振程度：
结论：

【3. 个股多窗口结构】
10日量价状态：
20日量价状态：
30日量价状态：
60日趋势背景：
120/240日大级别位置：

【4. 多日标签序列】
主要正常量价行为：
主要异常量价行为：
承接型标签数量：
供应型标签数量：
最近半窗口 vs 前半窗口变化：

【5. 综合结构判断】
当前阶段：
量价特点：
多空强弱：
是否与全A/板块共振：

【6. 后续确认】
看强确认：
看弱否定：
需要继续观察：

【7. 最终评级】
市场评分：
板块评分：
个股评分：
共振评分：
最终评级：
置信度：
```

## 13. 后效验证

系统必须支持后效验证。每个结构状态和序列模式都需要统计未来表现。

验证字段：

```text
future_ret_1d
future_ret_3d
future_ret_5d
future_ret_10d
future_ret_20d
future_max_gain_10d
future_max_drawdown_10d
future_max_gain_20d
future_max_drawdown_20d
hit_new_high_20d
hit_new_low_20d
outperform_sector_10d
outperform_market_10d
```

验证目标：

```text
1. 判断某类多日标签序列是否具有未来收益优势。
2. 判断某类异常标签组合是否能提前识别吸筹、派发、破位。
3. 判断全A/板块/个股共振是否显著提高成功率。
4. 判断逆全A或逆板块个股是否应降低评级。
```

## 14. 第一版实现范围

第一版只实现结构识别，不做自动交易。

必须实现：

```text
1. 读取全A、板块、个股日线数据
2. 计算百分比价格特征
3. 按不同窗口计算成交量相对强弱
4. 每个交易日生成多窗口量价标签
5. 用更大窗口判断趋势背景
6. 对多日标签序列做统计
7. 识别基础结构状态
8. 按全A -> 板块 -> 个股顺序评级
9. 导出 Excel
10. 支持后效验证字段
```

第一版不实现：

```text
1. 自动买卖
2. 实盘交易
3. 复杂机器学习
4. 主力行为确定性判断
5. 单日信号直接买入
```

## 15. 推荐目录结构

```text
modules/research/vpa_structure_recognizer/
│
├── README.md
├── SPEC.md
├── config.yaml
│
├── run_vpa_structure.py
├── sql/
│   ├── create_tables.sql
│   ├── load_daily_bars.sql
│   ├── build_market_bars.sql
│   ├── build_sector_bars.sql
│   └── export_report.sql
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── trend_context.py
│   ├── bar_labeler.py
│   ├── sequence_analyzer.py
│   ├── state_classifier.py
│   ├── top_down_ranker.py
│   ├── backtest_validator.py
│   └── excel_exporter.py
│
├── outputs/
│   ├── reports/
│   └── validation/
│
└── tests/
    ├── test_features.py
    ├── test_bar_labels.py
    ├── test_sequence_patterns.py
    └── test_state_classifier.py
```

## 16. config.yaml 示例

```yaml
windows:
  base: [10, 20, 30, 60, 120, 240]
  parent_map:
    10: [30, 60]
    20: [60]
    30: [120]
    60: [240]
    120: [240]
    240: [240]

volume_levels:
  low: 0.7
  normal_upper: 1.2
  mild_high: 1.8
  high: 2.5

price_position:
  low: 0.25
  mid_low: 0.45
  mid: 0.65
  mid_high: 0.85

label_thresholds:
  strong_body_ratio: 0.45
  long_shadow_ratio: 0.45
  close_high_position: 0.65
  close_low_position: 0.35
  high_volume_threshold: 1.5
  extreme_volume_threshold: 2.5

scoring:
  market_weight: 0.25
  sector_weight: 0.30
  stock_weight: 0.35
  resonance_weight: 0.10

rating:
  A: 80
  B: 65
  C: 50
  D: 35
```

## 17. 验收标准

系统完成后，必须满足以下条件：

```text
1. 同一股票同一交易日，能够生成不同 window_n 下的不同量价标签。
2. 所有价格振动、实体、影线、涨跌幅都使用百分比计算。
3. 成交量放量/缩量按当前 window_n 动态计算。
4. 单日标签不包含吸筹、派发、出货等主观定义。
5. 趋势背景使用更大窗口判断，而不是当前窗口自我判断。
6. 阶段判断必须基于多日标签序列，而不是单日标签。
7. 分析顺序必须是全A -> 板块 -> 个股。
8. 个股评级必须受到全A和板块状态影响。
9. 输出 Excel 至少包含市场、板块、个股、多窗口标签、后效验证五类结果。
10. 系统能够对 2022、2023、2024 年历史数据进行批量运行。
```

## 18. 一句话总结

这个系统的本质是：

```text
用百分比化价格波动和动态窗口成交量，给每个交易日生成上下文相关的多窗口量价标签；
再通过多日标签序列、父级趋势背景、全A与板块环境，自上而下识别股票当前所处的量价结构阶段。
```

最终它不是一个单日 K 线形态识别器，而是：

```text
A股全市场 -> 板块 -> 个股的多层级量价结构识别与筛选系统。
```
