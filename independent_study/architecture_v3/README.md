# 结构消融v3

本版只改网络竞争、阈值开关和层数。范围见[版本索引](../VERSIONS.md)与[协议](PROTOCOL.md)。输入、目标、编码、线性读出及23段滚动沿用v2，旧结果完整保留。

项目根目录使用现有环境：

```bash
export PYTHONPATH=independent_study
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=1
.venv/bin/python -m pytest independent_study/architecture_v3/tests -q
.venv/bin/python -m architecture_v3.run init
.venv/bin/python -u -m architecture_v3.run train --partition 0
.venv/bin/python -u -m architecture_v3.run train --partition 1
.venv/bin/python -u -m architecture_v3.run train --partition 2
```

三个train可分别并行启动，各负责fold编号模3的互斥目录。init完成后源码/协议/config锁定，不可静默修改并续跑。已完成段只在哈希正确时复用。先跑init再启动worker，避免并发初始化。

soft_fixed与soft_adapt是64神经元单层；deep_layer2是32神经元第二层，其第一层引用同种子的soft_adapt。deep_last使用第二层96维，deep_concat使用两层288维；两者共享神经模型，只读出不同。

每层checkpoint旁有训练锁；每个读出有验证后锁。samples含sample_id和fold，predictions的11列与v2相同；含_s104等的键是种子预测，去掉种子后是三个独立读出的等权平均。v2硬竞争与原始基线预测在相同样本上原样复制。
