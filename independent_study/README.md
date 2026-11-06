# IC分钟SNN/STDP独立研究

[版本索引](VERSIONS.md)区分已完成v1/v2和结构消融v3。v3训练命令见[说明](architecture_v3/README.md)；三个worker全部结束后，用现有环境执行`.venv/bin/python independent_study/analyze_v3.py`生成结构比较和报告（仍需`PYTHONPATH=independent_study`）。

最新[滚动训练与四组定位实验报告 v2](results/v2/report.md)及[复现入口](diagnostics_v2/README.md)已完成，包含冻结/只更新读出/全流程滚动、AB/BA合成对照、线性/非线性读出、当前与未来状态和连续资金回测。4—9月已观察，v2属于滚动历史诊断。

首轮结果保留在[完整中文报告 v1](results/v1/report.md)。以下命令和说明针对v1；[大纲](OUTLINE.md)和[预先固定的协议](PROTOCOL.md)也属于v1。旧research/仅参考流程，不导入代码；本目录没有依赖其实现或权重。

在仓库根目录运行：

```bash
export PYTHONPATH=independent_study
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
.venv/bin/python -m pytest independent_study/tests -q
.venv/bin/python -m iclab.run prepare --config independent_study/configs/experiment.json
.venv/bin/python -u -m iclab.run train --config independent_study/configs/experiment.json
.venv/bin/python -u -m iclab.run evaluate --config independent_study/configs/experiment.json
.venv/bin/python independent_study/render_report.py
```

现有v1已执行时不要覆盖：重跑需复制配置、更改output_dir，再向每阶段传同一配置。train拒绝覆盖已有选择锁，evaluate拒绝覆盖已完成评价。render_report.py只重建呈现和事后诊断，不改变模型或策略。

数据入口data/standardized/ic_v1/contracts/。先prepare产生按前日信息选合约的分钟表、因果特征、样本和标签，再train运行18模型、验证并写selection_lock.json；evaluate校验已锁文件哈希，冻结预测并回测。测试期不作在线学习，也不反复挑阈值。

读取实际输出：

```python
import numpy as np
import pandas as pd

base = "independent_study/results/v1/"
data = np.load(base + "dataset.npz")
X, y = data["X"], data["y"]  # (27150, 12, 12), (27150, 5)
samples = pd.read_parquet(base + "samples.parquet")
pred = pd.read_parquet(base + "test_predictions.parquet")
print(pred[["contract_id", "datetime", "stdp__return_5", "stdp__absolute_5"]].head())
trades = pd.read_csv(base + "backtests/stdp_trades.csv")
```

每个预测可用sample_id关联输入窗口，用bar_id/entry_bar_id/exit_bar_id追溯行情。收益输出单位bps；神经元活动存在representations/，每模型列顺序为64全窗发放率、64末3分钟发放率、64末态膜电位。

代码职责：data.py数据与特征；neural.py独立编码/LIF/成对STDP；analysis.py读出与统计；backtest.py逐bar事件和成本；run.py阶段驱动；render_report.py图表和中文报告；tests/机制及回测检查。

依赖复用工作区Python3.12环境，版本见requirements.txt。所有源输入只读，模型和报告写到配置output_dir。第三方仓库用于资料阅读，本研究神经实现使用PyTorch张量，不直接调用SpikingJelly或BindsNET。
