# 滚动训练与四组定位实验

[协议](PROTOCOL.md)固定数据、40/10/5滚动、三训练方式、时序正负对照、线性/RFF读出和状态定义。[结果报告](../results/v2/report.md)给出实际运行结果。4—9月已经观察，本轮是滚动历史检验。

从项目根目录使用现有环境：

```bash
export PYTHONPATH=independent_study
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=1
.venv/bin/python -m pytest tests independent_study/tests independent_study/diagnostics_v2/tests -q
.venv/bin/python -m diagnostics_v2.rolling prepare
.venv/bin/python -u -m diagnostics_v2.rolling synthetic
.venv/bin/python -u -m diagnostics_v2.rolling rolling
.venv/bin/python independent_study/analyze_v2.py
.venv/bin/python independent_study/render_v2_report.py
```

已准备的数据不能再次prepare覆盖。rolling可在配置、源代码和数据哈希一致时复用已完成段；预测前锁与完成文件分别保存。另做实验须改配置output_dir，保留旧版本，不能复用旧测试称作新确认。

读取一段预测：

```python
import numpy as np
import pandas as pd
p = "independent_study/results/v2/folds/fold_00/"
m = pd.read_parquet(p + "samples.parquet")
a = np.load(p + "predictions.npz")
prediction = a["rolling_all__stdp__linear"]
# 0:5分钟收益bps；1:绝对收益bps；2:实现波动bps
# 3:7 当前四状态分数；7:11 未来四状态分数
print(m[["sample_id", "contract_id", "datetime"]].head())
print(prediction[:3])
```

三个种子预测单独保存为stdp_s104等，stdp为平均读出预测。trade_*数组为已经按过去验证门槛过滤的方向分数，0表示不下单。当前状态标签只用于解释性诊断；未来状态标签不得加入交易输入。
