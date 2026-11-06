# v9 第一层事件调制资格迹

本版只比较第一层无反馈、真实事件反馈、错配事件反馈。主输出层始终使用真实价格事件的教师STDP。协议见 [PROTOCOL.md](PROTOCOL.md)，完成后结果见 [中文报告](../results/v9_eligibility/report.md)。

项目根目录，已有环境：

```bash
PYTHONPATH=independent_study OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q independent_study/eligibility_v9/tests independent_study/finance_v8/tests independent_study/sequence_v8/tests independent_study/event_stdp_v7/tests independent_study/risk_v4/tests
PYTHONPATH=independent_study OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m eligibility_v9.mechanism
PYTHONPATH=independent_study OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m eligibility_v9.train
PYTHONPATH=independent_study OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m eligibility_v9.analyze
PYTHONPATH=independent_study OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m eligibility_v9.trade
PYTHONPATH=independent_study OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m eligibility_v9.report
```

训练拒绝覆盖已存在结果。新增反馈是竞争诱发post的三因子辅助项，不是纯无监督学习、反向传播或在线逐分钟拟合。已成熟历史样本的重放保留v8随机训练顺序；预测冻结权重。具体衰减和反馈力度只检验这一预设，不代表所有资格迹算法。

合成样本在严格90%分位阈值下因模板并列而沉默；原结果完整保留，`mechanism_audit`另用既有1.3阈值和同隐藏权重解释该现象，不改金融配置。无反馈金融组须精确重现v8。分析与回测只写新v9目录。
