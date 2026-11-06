# v8 真实金融信号与交易应用

结果见[中文报告](../results/v8_finance/report.md)。本版接入真实分钟数据，完成23折新隐藏层/输出训练与90组条件回测；合成机制报告保留于v8_mechanism，v7及更早结果不覆盖。

在项目根目录使用已有环境：

```bash
PYTHONPATH=independent_study OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q independent_study/finance_v8/tests independent_study/sequence_v8/tests independent_study/event_stdp_v7/tests independent_study/risk_v4/tests
PYTHONPATH=independent_study OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m finance_v8.train
PYTHONPATH=independent_study OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m finance_v8.analyze
PYTHONPATH=independent_study OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m finance_v8.trade
PYTHONPATH=independent_study OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m finance_v8.report
```

train拒绝覆盖已有非空输出；新实验需另立输出目录与配置。analyze和trade可以独立核验已冻结模型，但会重写本版分析结果；不改训练预测和权重。源码、输入、日期、阈值及旧产物哈希保存在source_lock.json和各折lock.json。

- `core.py`：真实时段延迟、训练阈值、无事件教师输出、风险分档与执行延迟。
- `train.py`：3种子×23折，新模型/对照、过去验证标尺与预测。
- `analyze.py`：事件/提前量/配对区间及复载。
- `trade.py`：固定原方向机会的过滤/分档/择时，费用滑点和延迟敏感性、独立核账。
- `controls.py`：保留覆盖范围的严格同暴露随机控制。
- `report.py`：中文报告、预测和净值图、产物清单。

主分数不是多空方向或校准概率。训练隐藏层用竞争式STDP变体，事件教师只训练输出连接；autonomous对照使用自主IF放电，不用价格事件训练权重。所有结果仍属已观察历史上的探索性条件模拟，没有自动实盘部署。
