# v10 复现与读取

完整中文报告：[results/v10/report.md](../results/v10/report.md)。本轮是既有历史上的自主研究，未证明 STDP 独有增量或净盈利。

## 当前结果直接读取

- `../results/v10_refit/forecasts.parquet`：15900行，sample_id/bar_id/contract_id/datetime/trading_date/fold、realized（未来执行5分钟实现方差，bp²），以及raw40/raw50/readout50_stdp等正值方差预测。sqrt后才是bp波动尺度。
- `../results/v10_economics/base__readout50_stdp_opportunities.csv`：672个固定方向机会，lots是实际整数手数，含预测方差、现金波动预算、价格、费用、毛利、净利、同暴露期望风险等。lots=0的机会仍保留；不是672笔都成交。
- `../results/v10/`：综合指标、开发/审计拆分、状态解释、四幅图、报告与核验。
- `../results/v10_*/fold_XX/*.npz`：分折权重、状态概率q、读出参数；lock.json记录过去日期、来源和文件哈希。

## 运行环境

从项目根目录运行，使用现有 `.venv/bin/python`（Python 3.12，NumPy/Pandas/PyArrow/PyTorch/Matplotlib/Pytest）。无须安装上游SpikingJelly或旧TensorFlow e-prop环境。

```bash
export PYTHONPATH=independent_study
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

模型训练入口拒绝覆盖既有输出。需要完整重训时，在独立工作目录保留相同项目结构和冻结的v1/v2/v8/v9依赖，使用新的v10输出目录；不要删除本次结果来“更新”历史结论。阶段A读取旧网络，B—F复用真实合约特征和滚动日历。脚本中的ROOT由代码位置解析。

## 实际阶段顺序

```bash
.venv/bin/python -m research_v10.probe
.venv/bin/python -m research_v10.state_run develop
.venv/bin/python -m research_v10.state_run audit
.venv/bin/python -m research_v10.state_analyze
.venv/bin/python -m research_v10.increment
.venv/bin/python -m research_v10.modulated_run develop
.venv/bin/python -m research_v10.modulated_run audit
.venv/bin/python -m research_v10.temperature_run develop
.venv/bin/python -m research_v10.temperature_run audit
.venv/bin/python -m research_v10.refit_run develop
.venv/bin/python -m research_v10.select_candidate
.venv/bin/python -m research_v10.refit_run audit
.venv/bin/python -m research_v10.economics
```

设计记录依次为PROTOCOL.md、STAGE_B.md至STAGE_F.md、ECONOMIC_PROTOCOL.md。后续阶段是在前序发现基础上提出，不能把整套探索说成研究开始前就全部预注册。当前选择锁由开发结果生成，select_candidate会检查但不覆盖它；首次重新运行应在F审计之前创建。

以下分析命令不重训预测模型，只写入综合分析或核验文件：

```bash
.venv/bin/python -m research_v10.analyze_all
.venv/bin/python -m research_v10.verify_all
.venv/bin/python -m research_v10.figures
.venv/bin/python -m research_v10.report
.venv/bin/python -m pytest -q independent_study/research_v10/tests independent_study/risk_v4/tests independent_study/finance_v8/tests
```

verify_all依赖旧v8/v9及阶段A的冻结manifest，重算首/中/末折完整预测、检查所有日期/目标端点/分箱/验证选择和文件锁。state_analyze另核验阶段B。最后的汇总manifest由finalize.py生成或在已存在时只验证。

主模型使用16状态、3个种子；180维脉冲人口/延迟表示，36个输入活动，概率坐标局部STDP，推理连续状态强度。不是逐时步LIF或完整e-prop/HMM。输出层使用成熟未来方差标签，是混合学习。具体公式和所有阴性结果见报告。
