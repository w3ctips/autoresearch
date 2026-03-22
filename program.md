# autoresearch-futures

这是期货策略自主研究实验。

## Setup

1. 确认分支: `git checkout -b autoresearch/run-$(date +%Y%m%d)`
2. 读取文件:
   - `prepare.py` — 数据准备（固定）
   - `signals.py` — 信号规则（可修改）
   - `evolve.py` — 进化循环（可修改）
   - `backtest.py` — 回测引擎（固定）
3. 验证数据: `ls ~/.cache/autoresearch-futures/raw/`
   - 如无数据，运行: `python -m autoresearch_futures prepare`
4. 确认配置: 检查 `config.py` 中的参数

## 进化规则

**可以修改:**
- `signals.py` — 信号规则逻辑、参数阈值
- `evolve.py` 中的 `DEFAULT_PARAMS` — 各理论参数
- `evolve.py` 中的 `THEORY_WEIGHTS` — 理论权重

**禁止修改:**
- `prepare.py` — 数据准备逻辑固定
- `backtest.py` — 回测引擎固定
- `config.py` 中的数据划分参数（train_window_months 等）

**目标: 最大化验证集上的综合得分**

得分公式:
```
score = 0.25*夏普 + 0.30*扣费收益 + 0.15*胜率 + 0.15*精确率 - 0.15*最大回撤
```

## 实验循环

```
LOOP:
1. 修改 signals.py 或 evolve.py 中的参数/规则
2. git commit -m "experiment: 描述改动"
3. 运行: python -m autoresearch_futures evolve > run.log 2>&1
4. 提取结果: grep "^score:" run.log
5. 记录到 results.tsv
6. 如果得分提高: 保留提交
7. 如果得分降低: git reset --hard HEAD~1
8. 继续下一个实验
```

## 结果日志格式

results.tsv (TSV格式，不要提交到git):
```
split_id	commit	score	sharpe	net_return	win_rate	max_dd	description
001	abc123	0.850000	1.2400	0.1520	0.5800	0.0800	baseline
002	def456	0.920000	1.4500	0.1780	0.6200	0.0700	increase RSI period to 21
```

## 进化操作类型

| 操作 | 文件 | 示例 |
|------|------|------|
| 参数调整 | evolve.py | `"rsi_period": 14 → 21` |
| 阈值调整 | signals.py | `rsi_overbought: 70 → 75` |
| 信号组合 | signals.py | `signal = smc_signal & momentum_signal` |
| 周期切换 | evolve.py | `"timeframe": "1h" → "2h"` |
| 集成权重 | evolve.py | `"smc": 0.35 → 0.40` |

## 注意事项

- 简单优于复杂：相同表现时选择更简单的规则
- 永不停止：直到人工中断
- 不要在同一窗口上反复测试（过拟合风险）
- 记录每个实验的想法和结果