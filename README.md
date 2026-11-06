# SNN / STDP 金融时序研究

研究金融特征及其时序组合能否通过 SNN/STDP 提取增量预测信息，并用于股指期货的风险预测与交易风险控制。

当前已完成真实合约分钟数据整理、特征构建、多版本训练、滚动预测和含成本的条件回测。最新完成版本为 v10。**尚未建立稳定的 STDP 独有预测增量，也未建立扣除成本后的盈利策略。** 现有结果属于已经观察历史上的探索性研究。

## 阅读入口

- [版本索引](independent_study/VERSIONS.md)：各版本研究问题、状态和报告。
- [最新综合报告 v10](independent_study/results/v10/report.md)：风险预测、状态学习和风险预算结果。
- [v10 方法与运行说明](independent_study/research_v10/README.md)、[文献依据](independent_study/research_v10/RESEARCH.md)。
- [研究计划](task_plan.md)、[研究发现与文献笔记](findings.md)、[阶段进展](progress.md)。
- [项目长期记忆](AGENTS.md)：后续工作必须保留的目标、已知结果和研究边界。

## 仓库布局

```text
independent_study/       独立研究代码、配置、测试、各版本报告与结果统计
snn_data/               按真实合约整合原始分钟行情的代码
snn_quant/              早期基线实现，保留历史上下文
configs/                早期基线配置
tests/                  数据整理及早期基线测试
docs/                   数据结构与早期实验设计
AGENTS.md               项目长期记忆
task_plan.md            研究计划
findings.md             研究发现和文献笔记
progress.md             阶段记录
```

仓库保留 `independent_study/` 这一层目录，保持既有代码、配置和文档中的相对路径。所有命令均从仓库根目录运行。

## 环境与基础检查

原实验环境是 Linux、Python 3.12 和 PyTorch 2.6 CPU。独立研究依赖版本见 [requirements](independent_study/requirements.txt)。

```bash
git clone https://github.com/selinaxian/independent_study.git
cd independent_study
python3.12 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r independent_study/requirements.txt
export PYTHONPATH=independent_study
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
.venv/bin/python -m pytest -q independent_study/tests
```

此基础检查使用合成数据，不代表已经重现真实行情上的模型结果。运行具体版本前请阅读其 README 和协议；已有冻结实验不能直接覆盖。

## 数据和完整复现

原始行情位于本地 `data/分钟线/`，标准化合约表位于 `data/standardized/ic_v1/contracts/`。字段包含合约代码、时间、OHLC、成交量、持仓量及来源追踪，详见 [数据格式](docs/data_layout.md)。数据整理入口是 `python -m snn_data.standardize`，读取 7z 还需要系统 `libarchive`。

**本仓库上传代码、配置、报告、图表、CSV 统计和 JSON 审计记录。原始行情、Parquet、模型检查点、密集数组、运行环境和第三方克隆保留在本地，未上传。** GitHub 不是完整实验文件的备份。

因此，克隆仓库可以阅读和检查实现，但金融重训、模型复载核验及部分报告中的产物链接还需要对应的本地数据和检查点。哈希清单保留完整实验的记录，不表示清单列出的所有文件均在 GitHub。

首次重现请按 [v1 入口](independent_study/README.md) 和 [版本索引](independent_study/VERSIONS.md) 准备依赖；v10 依赖先前版本生成的特征、滚动日历和部分模型文件，不能只运行最后一个脚本。原 `research/` 和第三方参考仓库不纳入此仓库。

## 后续同步

在本地项目根目录执行：

```bash
git status
git add <本次修改的文件或目录>
git commit -m "说明本次研究或文档更新"
git push
```

修改文件后需要提交并推送才会更新 GitHub；没有设置自动上传。原始数据和大体积产物由 `.gitignore` 排除，仍需单独备份。
