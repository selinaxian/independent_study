# v10阶段A：冻结隐藏层、统一读出定位

完整23折/3种子，共575个Ridge读出；没有重训隐藏层，也未更改数据和事件定义。所有15540预测bar_id与标签对齐v9，原输入Ridge预测复现误差≤1e-12。

真实反馈隐藏脉冲经相同Ridge读出，AUC从原教师输出0.4767恢复到0.5242，差0.04747，95%五日块[0.00887,0.08663]；阈值前驱动约0.5540，仍低于原120维输入0.5760。已有表示存在可恢复的信息，不能继续只归因第一层完全学不到；但真实反馈并没有超过无反馈/错配，同样的随机线性投影drive约0.5535，也不能把连续驱动读出改善全归功于STDP。

无反馈/真实反馈spike读出相对随机spike有优势，但均较弱且不胜输入；这项有限正面证据保留，不以最终经济失败抹去。随机驱动与随机阈值spike的差异也说明阈值化和活动分布影响显著。

所有读出使用相同训练标准化、alpha网格和验证Brier选择。表中Brier仅适用于监督读出，其余原始教师/波动分数不是概率，保留NaN。AUC/AP先按过去验证分位校准。历史已被观察，全部仍属探索性定位。

|model|n|auc|ap|brier|
|---|---|---|---|---|
|input_ridge|15540|0.576019|0.132907|0.092155|
|mismatched_drive|15540|0.556187|0.121976|0.093031|
|mismatched_spike|15540|0.536079|0.119809|0.093311|
|original_current_rv|15540|0.668824|0.186887|nan|
|original_mismatched|15540|0.475899|0.099025|nan|
|original_true_feedback|15540|0.476721|0.099265|nan|
|original_unsupervised|15540|0.475593|0.099162|nan|
|random_drive|15540|0.553494|0.119008|0.093005|
|random_spike|15540|0.422038|0.090308|0.093458|
|true_feedback_drive|15540|0.553980|0.123110|0.093050|
|true_feedback_spike|15540|0.524188|0.112490|0.093347|
|unsupervised_drive|15540|0.547597|0.119835|0.093138|
|unsupervised_spike|15540|0.531825|0.116606|0.093323|

|left|right|metric|difference|lower|upper|
|---|---|---|---|---|---|
|true_feedback_spike|original_true_feedback|auc|0.047467|0.008875|0.086631|
|true_feedback_spike|original_true_feedback|ap|0.013225|0.002153|0.027208|
|true_feedback_spike|random_spike|auc|0.102150|0.000056|0.205774|
|true_feedback_spike|random_spike|ap|0.022182|0.000417|0.042283|
|unsupervised_spike|random_spike|auc|0.109787|0.008543|0.215048|
|unsupervised_spike|random_spike|ap|0.026298|0.004236|0.047729|
|true_feedback_drive|input_ridge|auc|-0.022039|-0.039089|-0.006888|
|true_feedback_drive|input_ridge|ap|-0.009797|-0.022139|0.000290|
|true_feedback_spike|input_ridge|auc|-0.051831|-0.068229|-0.035341|
|true_feedback_spike|input_ridge|ap|-0.020417|-0.032330|-0.009803|

下一阶段依据：[概率状态风险框架协议](../../research_v10/STAGE_B.md)。本阶段没有新增策略回测或盈利结论。
