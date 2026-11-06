# v10 文献、源码与设计判断

用户于2026-10-29授权独立探索，不再限定已有老师建议。以下明确区分论文结论、源码事实和本项目推断，金融有效性必须另行验证。

## 已核对的研究路线

- Nessler et al. (2013), [Bayesian Computation Emerges in Generic Cortical Microcircuits through STDP](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003037)。理论条件下，特定权重依赖STDP与WTA/兴奋性适配可学习输入的概率模型。启发：把神经元解释为市场状态，而非默认发放次数就是风险；不能把任意竞争LTP实现都称为该论文算法。
- Kappel et al. (2014), [STDP Installs in WTA Circuits an Online Approximation to HMM Learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC3967926/)。加入横向兴奋性连接与STDP，学习状态转换；无奖励的在线近似与完整HMM学习能力有区别。启发：状态概率及状态转移可以有金融解释，不必强求某单元天然预测涨跌。不同延迟特征的静态聚类不能冒充完整HMM。
- Bellec et al. (2020), [A solution to the learning dilemma for recurrent networks of spiking neurons](https://pmc.ncbi.nlm.nih.gov/articles/PMC7367848/)，[官方代码](https://github.com/IGITUGraz/eligibility_propagation)。e-prop把目标误差学习信号与局部资格迹结合，近似梯度学习；不是普通无监督STDP。启发：正确的信用分配需要动力学相关资格迹和有效目标信号，v9的单窗口小辅助项不能代表这种完整方法。
- Corsi (2009), [A Simple Approximate Long-Memory Model of Realized Volatility](https://doi.org/10.1093/jjfinec/nbp001)。用不同时间尺度的实现波动构建简洁基线。分钟5/10/20窗口只能称HAR启发的短周期多尺度基线，不能称原论文日/周/月模型原样复现。

## 本地源码核对

- `bindsnet/bindsnet/learning/learning.py:MSTDPET`包含pre/post迹、时间推进的eligibility及eligibility_trace、reward调制；默认时步延迟和zero_lag有区别。本轮先读源码，没有把现有依赖整体升级或把演示当金融基准。
- `spikingjelly/spikingjelly/activation_based/learning/plasticity.py`提供STDPLearner/MSTDPLearner/MSTDPETLearner；对应examples/plasticity/mstdpet.py明确是机制demo，随机奖励不是任务验证。旧learning.py文件路径不存在，现为learning包。

## 初始判断与阶段顺序

1. 先做冻结隐藏层、统一监督读出的定位。v9只挑一个训练期最佳单元的探针不足以排除32维联合信息；必须在同一输入、标签、滚动日期、损失/正则化选择下比较。
2. 金融主要目标优先考虑可执行5分钟持有窗口的连续风险及其增量，而不是将所有未来风险压成一个稀疏绝对阈值事件。保留事件任务作交叉验证，不按结果随意换目标。
3. 新框架候选为“金融风险底座+STDP状态/时序表示+增量读出”。必须有匹配随机网络、无STDP/乱序及同信息原始特征对照；不能仅凭混合模型赚钱就把贡献归功于STDP。
4. 连续风险应评价校准/损失、相对基线改进及稳定性；IC不能替代风险估计准确度。经济评价优先检验风险预算/尾部暴露控制，保留交易费用；不把原低毛利方向策略作为唯一成功标准。
5. 历史2026年4—9月已多次查看。所有新增历史结果标为探索性；设置仅开发段搜索、冻结后历史审计，可减轻本轮选择偏差但不能使旧日期重新未见。

本页持续记录实际查到的依据、实现差异与失败实验，不因结果不理想删去路线。

补充核对：Nessler论文Results/式5以post触发权重依赖LTP，近期没有pre的连接发生LTD，其概率解释与特定输入/权重约束有关；不能只增强活动pre后归一化就认为等价。第一个PMC入口出现浏览器验证，改从PLOS原刊读取。其他PMC页面后续也出现验证，未绕过访问限制。
Patton (2011), [Volatility forecast comparison using imperfect volatility proxies](https://www.sciencedirect.com/science/article/pii/S030440761000076X)，讨论代理波动噪声下预测损失的稳健性。采用QLIKE必须说明方差代理与条件假设；它不能自动消除数据质量、目标错配或有限样本问题。
本地Git版本：BindsNET c7ac73a37b3c08bb351bb9fb163e4720c61d9081；SpikingJelly 1b4b76b2a1acc59cfdb8ab716d3c8ea8ccda0969。官方e-prop仓库浅克隆供源码学习，暂不安装运行旧TensorFlow依赖。

## 阶段A结果
23折/3种子/575统一Ridge读出完成。原真实反馈输出AUC .476721，同隐藏spike→Ridge .524188、drive→Ridge .553980；原输入→Ridge .576019。无反馈spike .531825，错配spike .536079，随机spike .422038；随机drive .553494。说明旧教师输出存在可恢复的信息，但目前学习层仍没有超过原输入；真实反馈没有成为最好，线性drive的表现和随机投影接近。不得仅凭替换读出后变好，就宣称STDP有金融优势。
官方e-prop源码已浅克隆，commit efd02e6879c01cda3fa9a7838e8e2fd08163c16e，说明Python3.6/3.7与TensorFlow1.14/1.15，含解析资格迹与自动微分梯度的核验。该源码学习不等于已运行e-prop金融模型。

阶段B实现差异固定：使用概率坐标q的post触发更新q+=eta*(近期pre-q)，w=log q，其一阶形式对应权重依赖LTP与缺失pre抑制；明确是理想化变体而非论文全文复现。每观察窗重置pre，0/2/4分钟延迟负责顺序，未实现递归HMM。推理读归一化状态强度，避免最终只剩总发放量。
