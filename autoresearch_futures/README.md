# autoresearch-futures

中国期货 15min 高频自进化策略研究系统。

将 autoresearch 框架改造为期货策略自主研究系统，实现：
- 多理论（SMC、动能、线性推演）分别进化，组合成多策略系统
- 产出实盘交易信号并推送

## 功能特性

- **三大信号理论**: SMC结构分析、动能指标、线性推演
- **动态周期合成**: 从15min基础周期合成30min/1h/2h/4h
- **Walk-Forward验证**: 12月训练 + 2周Embargo + 1月验证，防止过拟合
- **多目标评估**: 扣费净收益 + 夏普比率 + 胜率 + 信号精确率
- **信号推送**: 微信/Telegram/邮件推送交易信号

## 项目结构

```
autoresearch_futures/
├── __main__.py        # CLI 入口
├── config.py          # 配置模块
├── prepare.py         # 数据准备模块
├── signals.py         # 信号规则模块 (SMC/动能/线性推演)
├── backtest.py        # 回测引擎
├── ensemble.py        # 多理论集成
├── notify.py          # 信号推送
├── evolve.py          # 进化主循环
└── pyproject.toml     # 依赖配置
```

## 快速开始

### 1. 创建虚拟环境并安装依赖

```bash
# 进入项目目录
cd /Users/maxclaw/Documents/github/autoresearch/autoresearch_futures

# 创建虚拟环境 (需要 Python 3.10+)
python3 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 安装项目依赖
pip install -e ".[dev]"
```

### 2. 准备数据

```bash
# 回到项目根目录
cd ..

# 下载期货数据并生成滚动窗口切分
python -m autoresearch_futures prepare
```

这会：
- 使用 akshare 下载全市场主力合约的 15min K线数据
- 合成 30min/1h/2h/4h 周期数据
- 生成 Walk-Forward 滚动窗口切分

数据存储在 `~/.cache/autoresearch-futures/`

### 3. 运行进化实验

```bash
# 运行单次进化实验
python -m autoresearch_futures evolve
```

输出示例：
```
---
score:    0.851234
rb: sharpe=1.2400 return=0.1520
i: sharpe=1.1800 return=0.1340
```

### 4. 运行测试

```bash
python -m autoresearch_futures test
# 或
pytest tests/ -v
```

## 自主研究模式

让 AI Agent 读取 `program.md` 开始自主研究：

### 方式一：启动新的 Claude Code 会话

```bash
cd /Users/maxclaw/Documents/github/autoresearch
claude
```

然后输入：
```
请读取 program.md 文件，按照其中的 Setup 步骤准备好环境，然后开始自主研究循环。
```

### 方式二：直接提示 Agent

在你的 Claude Code 会话中输入：

```
读取 program.md，开始一个新的实验
```

Agent 会自动：
1. 创建 `autoresearch/run-YYYYMMDD` 分支
2. 读取所有相关文件
3. 开始进化实验循环
4. 记录结果到 `results.tsv`

## CLI 命令

```bash
# 数据准备
python -m autoresearch_futures prepare [--symbols rb i hc] [--force]

# 运行进化
python -m autoresearch_futures evolve [--symbols rb i]

# 运行测试
python -m autoresearch_futures test
```

## 配置说明

核心配置在 `config.py` 中：

```python
# 数据划分参数
train_window_months = 12      # 训练窗口
embargo_weeks = 2             # 隔离期
valid_window_months = 1       # 验证窗口
locked_predict_months = 6     # 锁定预测集

# 交易成本
commission_rate = 0.0001      # 手续费率 (万分之一)
slippage_ticks = 1            # 滑点 tick 数

# 理论权重
THEORY_WEIGHTS = {
    "smc": 0.35,
    "momentum": 0.35,
    "linear": 0.30,
}
```

## 信号理论说明

| 理论 | 指标/模式 | 适用周期 |
|------|----------|---------|
| SMC | Order Block, FVG, 流动性扫荡 | 1h |
| 动能 | RSI, MACD, 价格速度 | 15min |
| 线性推演 | 趋势检测, 回归通道, 突破 | 30min |

## 得分公式

```
score = 0.25*夏普比率 + 0.30*扣费收益 + 0.15*胜率 + 0.15*信号精确率 - 0.15*最大回撤
```

## 依赖

- Python 3.10+
- akshare (期货数据源)
- pandas, numpy (数据处理)
- scipy (统计计算)
- requests (HTTP请求)

## License

MIT