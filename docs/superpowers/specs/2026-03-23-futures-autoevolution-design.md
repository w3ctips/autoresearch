# 中国期货 15min 高频自进化策略研究系统设计文档

## 概述

将 autoresearch 框架改造为期货策略自主研究系统，实现：
1. 多理论（SMC、动能、线性推演）分别进化，组合成多策略系统
2. 产出实盘交易信号并推送

## 目标

| 维度 | 决策 |
|------|------|
| 目标 | 多策略组合 + 实盘信号推送 |
| 数据源 | akshare，全市场期货品种 |
| 理论模块 | SMC结构分析、动能指标、线性推演 |
| K线周期 | 15min 基础，动态合成更大周期 |
| 数据划分 | Purged Walk-Forward + 锁定预测集 |
| 评估指标 | 扣费净收益 + 多目标加权 |
| 进化机制 | 分层进化（规则→参数→集成） |
| 进化周期 | 12月训练 + 2周Embargo + 1月验证 |
| 信号输出 | 微信/Telegram/邮件推送 |
| 运行环境 | 本地服务器 24小时 |

## 整体架构

```
autoresearch-futures/
├── prepare.py          # 数据准备：akshare下载、K线合成、数据集划分
├── evolve.py           # 进化主入口：迭代循环、评估、保留/淘汰决策
├── signals.py          # 信号规则库：SMC/动能/线性三大理论的信号生成函数
├── backtest.py         # 回测引擎：扣费计算、风险指标、多品种并行回测
├── ensemble.py         # 多理论集成：信号加权、投票机制
├── notify.py           # 信号推送：微信/Telegram/邮件通知
├── config.py           # 全局配置：周期参数、成本参数、推送配置
├── results.tsv         # 进化结果日志（不提交git）
└── program.md          # agent 自主研究指令
```

**核心数据流：**

```
akshare → prepare.py → 本地数据缓存
                           ↓
evolve.py ← signals.py ← backtest.py
     ↓
ensemble.py → 最终信号 → notify.py → 推送
```

**Agent 修改目标：**
- `evolve.py` — 进化策略、集成权重
- `signals.py` — 信号规则逻辑

## 数据准备模块 (prepare.py)

### 核心常量

```python
BASE_TIMEFRAME = "15min"           # 基础周期
SYNTHETIC_TIMEFRAMES = ["30min", "1h", "2h", "4h"]  # 可动态合成的周期

TRAIN_WINDOW_MONTHS = 12           # 训练窗口
EMBARGO_WEEKS = 2                  # 隔离期
VALID_WINDOW_MONTHS = 1            # 验证窗口
LOCKED_PREDICT_MONTHS = 6          # 锁定预测集（永不参与进化）

COMMISSION_RATE = 0.0001           # 手续费率（万分之一）
SLIPPAGE_TICKS = 1                 # 滑点tick数
```

### 数据目录结构

```
~/.cache/autoresearch-futures/
├── raw/                    # 原始数据（按品种存储）
│   ├── rb.parquet         # 螺纹钢
│   ├── i.parquet          # 铁矿石
│   └── ...
├── synthetic/              # 合成周期数据
│   ├── rb_30min.parquet
│   ├── rb_1h.parquet
│   └── ...
└── splits/                 # 数据集划分索引
    └── walk_forward.json  # 每个滚动窗口的时间边界
```

### 关键函数

```python
# 市场数据获取
download_all_contracts()       # 下载全市场主力合约
update_daily()                 # 每日增量更新

# K线合成
synthesize_timeframe(df, target_tf)  # 将15min合成为更大周期

# 数据集划分
generate_walk_forward_splits()  # 生成滚动窗口索引
get_train_data(split_id)        # 获取训练集
get_valid_data(split_id)        # 获取验证集
get_locked_predict_data()       # 获取锁定预测集
```

**品种过滤：** 自动排除上市不足 18 个月的新品种（训练窗口 + 验证窗口）

## 信号规则模块 (signals.py)

### 三大理论信号结构

```python
def smc_signals(df, params) -> dict:
    """
    SMC结构分析信号
    输入: K线数据 + 参数
    输出: {
        "order_block": Series,      # OB位置标记
        "fvg": Series,              # 公平价值缺口
        "liquidity_sweep": Series,  # 流动性扫荡
        "bos": Series,              # 结构突破
        "signal": Series,           # 综合信号 (-1, 0, 1)
    }
    """

def momentum_signals(df, params) -> dict:
    """
    动能指标信号
    输出: {
        "rsi": Series,
        "macd": Series,
        "macd_signal": Series,
        "price_velocity": Series,   # 价格速度
        "signal": Series,
    }
    """

def linear_signals(df, params) -> dict:
    """
    线性推演信号
    输出: {
        "trend": Series,            # 趋势方向
        "regression_band": Series,  # 回归通道
        "breakout": Series,         # 突破信号
        "signal": Series,
    }
    """
```

### 参数结构

```python
DEFAULT_PARAMS = {
    "smc": {
        "ob_lookback": 20,          # OB回溯周期
        "fvg_min_size": 0.002,      # FVG最小尺寸（价格比例）
        "sweep_threshold": 0.01,    # 扫荡阈值
        "timeframe": "1h",          # SMC适用周期
    },
    "momentum": {
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "timeframe": "15min",       # 动能适用周期
    },
    "linear": {
        "regression_period": 20,
        "band_std": 2.0,            # 回归通道标准差
        "breakout_confirm": 3,      # 突破确认K线数
        "timeframe": "30min",       # 线性推演适用周期
    },
}
```

**动态周期处理：** 每个信号函数内部根据 `timeframe` 参数调用对应周期的数据。

## 回测引擎模块 (backtest.py)

### 核心函数

```python
def run_backtest(signals, data, config) -> BacktestResult:
    """单品种回测"""

def run_multi_backtest(signals_dict, data_dict, config) -> dict:
    """多品种并行回测"""
```

### 交易成本计算

```python
def calc_trade_cost(trade, config):
    commission = trade.value * config.COMMISSION_RATE      # 手续费
    slippage = trade.volume * config.SLIPPAGE_TICKS * tick_size  # 滑点
    return commission + slippage
```

### 评估指标

```python
@dataclass
class BacktestResult:
    # 收益指标
    net_return: float          # 扣费净收益率
    annual_return: float       # 年化收益

    # 风险指标
    max_drawdown: float        # 最大回撤
    volatility: float          # 收益波动率
    var_95: float              # 95% VaR

    # 风险收益指标
    sharpe_ratio: float        # 夏普比率
    calmar_ratio: float        # 卡玛比率
    sortino_ratio: float       # 索提诺比率

    # 交易统计
    total_trades: int          # 总交易次数
    win_rate: float            # 胜率
    profit_factor: float       # 盈亏比
    avg_holding_bars: float    # 平均持仓K线数

    # 信号质量
    signal_precision: float    # 信号精确率
    signal_recall: float       # 信号召回率

    # 综合得分
    score: float               # 加权综合得分
```

### 综合得分公式

```python
def calc_score(result: BacktestResult, weights: dict) -> float:
    """
    综合得分 = w1*夏普 + w2*扣费收益 + w3*胜率 + w4*信号精确率 - w5*最大回撤
    """
    return (
        weights["sharpe"] * result.sharpe_ratio +
        weights["net_return"] * result.net_return +
        weights["win_rate"] * result.win_rate +
        weights["precision"] * result.signal_precision -
        weights["drawdown"] * result.max_drawdown
    )
```

默认权重：
```python
DEFAULT_WEIGHTS = {
    "sharpe": 0.25,
    "net_return": 0.30,
    "win_rate": 0.15,
    "precision": 0.15,
    "drawdown": 0.15,
}
```

## 进化主循环模块 (evolve.py)

### 核心流程

```python
def evolution_loop():
    while True:
        # 1. 获取当前状态
        current_split = get_current_split()

        # 2. 修改信号规则/参数（由 agent 或进化算法执行）

        # 3. 在训练集上生成信号
        train_data = get_train_data(current_split)
        signals = generate_all_signals(train_data, PARAMS)

        # 4. 在验证集上回测评估
        valid_data = get_valid_data(current_split)
        results = run_multi_backtest(signals, valid_data, CONFIG)

        # 5. 计算综合得分
        score = aggregate_scores(results)

        # 6. 记录结果
        log_results(current_split, score, results)

        # 7. 决策：保留或回滚
        if score > best_score:
            commit_changes()
            best_score = score
        else:
            rollback_changes()

        # 8. 滚动到下一个窗口
        advance_split()
```

### Git 分支策略

```
master                          # 稳定版本
└── autoresearch/run-20260323   # 单次进化运行分支
    ├── commit A (baseline)
    ├── commit B (improved, kept)
    ├── commit C (worse, reverted)
    └── commit D (improved, kept)
```

### 结果日志格式

```tsv
split_id	commit	score	sharpe	net_return	win_rate	max_dd	description
001	a1b2c3d	0.85	1.24	0.152	0.58	0.08	baseline
002	b2c3d4e	0.92	1.45	0.178	0.62	0.07	increase RSI period to 21
```

### 进化操作类型

| 操作类型 | 说明 | 示例 |
|---------|------|------|
| 参数调整 | 修改 DEFAULT_PARAMS 中的值 | `rsi_period: 14 → 21` |
| 信号组合 | 添加/移除信号条件 | `signal = smc_signal & momentum_signal` |
| 阈值调整 | 修改信号触发阈值 | `rsi_overbought: 70 → 75` |
| 周期切换 | 修改理论适用周期 | `smc_timeframe: 1h → 2h` |
| 集成权重 | 修改理论投票权重 | `smc_weight: 0.4 → 0.5` |

## 多理论集成模块 (ensemble.py)

### 集成策略

```python
def ensemble_signals(smc_signal, momentum_signal, linear_signal, weights) -> Series:
    """多理论信号集成 - 加权投票"""
    combined = (
        weights["smc"] * smc_signal +
        weights["momentum"] * momentum_signal +
        weights["linear"] * linear_signal
    )

    final_signal = np.where(combined >= 0.5, 1,
                   np.where(combined <= -0.5, -1, 0))

    return final_signal
```

### 集成模式

```python
# 模式 A：简单投票
def simple_vote(signals, weights) -> Series:
    """加权投票，超过阈值则发出信号"""

# 模式 B：一致性过滤
def consensus_filter(signals) -> Series:
    """只有多个理论同向时才发出信号"""

# 模式 C：层级触发
def hierarchical_trigger(signals, priority) -> Series:
    """按优先级分层，高优先级理论先检查"""
```

### 理论权重

```python
THEORY_WEIGHTS = {
    "smc": 0.35,
    "momentum": 0.35,
    "linear": 0.30,
}
```

### 信号置信度

```python
@dataclass
class SignalWithConfidence:
    direction: int           # -1, 0, 1
    confidence: float        # 0.0 - 1.0
    source_theories: list    # 贡献该信号的理论
    timestamp: datetime
```

## 信号推送模块 (notify.py)

### 推送渠道配置

```python
@dataclass
class NotifyConfig:
    wechat_enabled: bool = False
    wechat_webhook: str = ""

    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    email_enabled: bool = False
    email_smtp: str = ""
    email_sender: str = ""
    email_password: str = ""
    email_recipients: list = []
```

### 信号消息格式

```
【期货信号提醒】
品种: rb2405
方向: 做多
时间: 2026-03-23 14:30:00
价格: 3650.0
置信度: 75%

理论依据:
- SMC: Order Block 支撑
- 动能: RSI 超卖反弹

建议止损: 3620
建议止盈: 3720
```

### 信号冷却机制

```python
SIGNAL_COOLDOWN_MINUTES = 30  # 防止同一品种短时间内重复推送
```

## 数据划分方案

采用 **Purged Walk-Forward + 锁定预测集**：

```
|--- 训练窗口 12个月 ---|-- Embargo 2周 --|-- 验证 1个月 --|...滚动...|==== 锁定预测集 6个月 ====|
```

**核心原理：**
1. Walk-Forward 滚动验证：模拟真实交易
2. Embargo Period：防止 label leakage
3. 锁定预测集：永不参与进化，仅最终评估

## 进化机制

采用 **分层进化方案**：

```
第一层：信号规则进化（遗传规划）
├── SMC理论模块 → 生成信号规则组合
├── 动能理论模块 → 生成信号规则组合
└── 线性推演模块 → 生成信号规则组合

第二层：参数优化（贝叶斯优化）
└── 对第一层选出的规则进行参数精细调优

第三层：集成投票
└── 多理论信号加权组合，生成最终买入信号
```

## Agent 指令文件 (program.md)

```markdown
# autoresearch-futures

这是期货策略自主研究实验。

## Setup

1. 确认分支: `git checkout -b autoresearch/run-<date>`
2. 读取文件: prepare.py, signals.py, evolve.py, backtest.py
3. 验证数据: 检查 ~/.cache/autoresearch-futures/ 是否有数据
4. 确认配置: 检查 config.py 中的参数

## 进化规则

**可以修改:**
- `signals.py` — 信号规则逻辑
- `evolve.py` — 参数配置、集成权重

**禁止修改:**
- `prepare.py` — 数据准备逻辑固定
- `backtest.py` — 回测引擎固定
- `config.py` 中的数据划分参数

**目标: 最大化验证集上的综合得分**

## 实验循环

LOOP:
1. 修改 signals.py 或 evolve.py
2. git commit
3. 运行: `python evolve.py > run.log 2>&1`
4. 提取结果: `grep "^score:" run.log`
5. 记录到 results.tsv
6. 得分提高: 保留 | 得分降低: 回滚

## 注意事项

- 简单优于复杂
- 永不停止直到人工中断
```