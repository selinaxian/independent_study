# 延迟隐藏层：v8 合成机制核验

当前完成的是合成机制实验，金融训练和回测尚未运行。结构与教学说明见[报告](../results/v8_mechanism/report.md)，精确学习规则和范围见[设计](DESIGN.md)。

```bash
PYTHONPATH=independent_study OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q independent_study/sequence_v8/tests independent_study/event_stdp_v7/tests
PYTHONPATH=independent_study OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m sequence_v8.run
PYTHONPATH=independent_study OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m sequence_v8.report
```

训练拒绝覆盖非空results/v8_mechanism。另做实验应另建输出目录和配置，保留本轮证据。代码不读取research/、不读取金融行情、不修改v7；只复用v7教师成对STDP和指标函数。

主模型：24通道×5延迟→32窗口独立IF单元→32条教师STDP输出突触→一个分数。隐藏采用竞争诱发post与归一化STDP变体，不是自由运行LIF训练；价格事件只教输出突触，没有回传到隐藏连接。

隐藏矩阵以`channel * 5 + delay`索引输入，检查点包含训练权重、同初始化随机权重和活动量匹配阈值。learned_patterns.json里的示例单元选择仅用于事后解释，模型预测使用全部隐藏单元。
