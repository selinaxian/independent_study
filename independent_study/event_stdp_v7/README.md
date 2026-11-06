# v7 价格事件教师 STDP

本版将真实市场价格事件作为训练后突触教师脉冲，不是纯无监督隐藏层STDP。先读[固定协议](PROTOCOL.md)，结果见[中文报告](../results/v7/report.md)。原v1—v5保持不变，v6暂缓且未完成。

项目根目录运行，使用现有环境，无需安装依赖：

```bash
PYTHONPATH=independent_study OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q independent_study/event_stdp_v7/tests
PYTHONPATH=independent_study OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m event_stdp_v7.run
PYTHONPATH=independent_study OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m event_stdp_v7.analyze
```

训练入口会拒绝覆盖非空results/v7；新实验请修改输出位置并另立配置/协议，保留已有产物。分析入口会重算指标和报告并核对23折模型与输入哈希，不改变训练权重或预测。

- `core.py`：阈值穿越、因果事件、真实分钟trace、教师STDP、漏积分分数。
- `run.py`：23折滚动，等权、错配教师、当前波动和同输入Ridge对照。
- `analyze.py`：合成检查、完整复载、5日块配对区间、中文报告。
- `results/v7/folds/`：每折权重、特征/价格阈值、验证分数标尺及日期锁。
- `predictions.parquet`：每个模型/时点的原始分数、历史验证排序分位、告警和事后未来事件标签。
- `weights.csv`：每折每通道权重及正/反配对统计；`event_leads.csv`：含漏报的事件级提前量。

原型输出连续膜电位和阈值告警，没有自由运行发放后重置的LIF。主模型的突触由STDP直接学习，Ridge只是评价对照。24个输入通道、0个隐藏层、1个漏积分预警输出；未来事件只进入训练教师和事后评价。
