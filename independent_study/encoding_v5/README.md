# 复现v5

在项目根目录使用现有`.venv`，无需新依赖。必须保留v1/v2输入与冻结结果；v3/v4用于完整性核验。

```bash
PYTHONPATH=independent_study OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=1 .venv/bin/python -m encoding_v5.run init
PYTHONPATH=independent_study OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=1 .venv/bin/python -m encoding_v5.run run
PYTHONPATH=independent_study OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=1 .venv/bin/python -m encoding_v5.synthetic
PYTHONPATH=independent_study OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=1 .venv/bin/python independent_study/analyze_v5.py
```

若按折并行，`run`可加`--workers 2 --worker 0`和`--workers 2 --worker 1`；初始化只执行一次，两进程写不同fold目录。旧模型/源代码保持只读；新增完整fold可复用，改参数/代码应另建实验目录，不覆盖冻结记录。

首次脉冲时间决定于截至该市场分钟的幅度；不要据此解释分钟内市场因果。[协议](PROTOCOL.md)规定比较边界，报告必须区分编码重建精度、预测信息和STDP增量。
