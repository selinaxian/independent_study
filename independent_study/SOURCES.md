# 资料与使用边界

- Gao、Luk、Weston (2021)，本地reference/High-Frequency+Trading+SNN.pdf。原文用VWAP价格差分正负通道、Poisson、两组64隐藏神经元及单输出LIF，前日训练次日评价；输出脉冲用于择时，另用方向规则交易。原文已有前瞻窗口、随机择时、LSTM/ARIMA比较。本项目不声称精确复现；分钟OHLC数据不能重建逐笔VWAP。
- SpikingJelly commit 1b4b76b2a1acc59cfdb8ab716d3c8ea8ccda0969，activation_based/functional/learning.py的stdp_linear_step：阅读迹与成对更新。本项目独立实现，采用指数衰减、批平均、排除零时间差，和该函数更新顺序/批求和不同。
- BindsNET commit c7ac73a37b3c08bb351bb9fb163e4720c61d9081，learning/README.md区分PostPre与DiehlCook2015，作为方法分类参考。snntorch与Shikhargupta仓库仅用于LIF、编码、竞争及阈值机制理解，不导入其实现。
- [中金所中证500股指期货合约细则](https://www.e-cffex.com.cn/u/cms/www/202001/17094604ed9c.pdf)：乘数200元/点、最小变动0.2点、第三个周五交割月到期等。节假日顺延并未在本项目逐合约核实，使用提前3日排除规则。
- [中金所2023年第16号通知](https://www.cffex.com.cn/jystz/20230317/31868.html)，[建信期货转载](https://www.ccbfutures.com/main/a/20230317/60122.shtml)：2023-03-20起IC等股指平今费率调整至成交金额万分之2.3。
- [东海期货关于股指交易收费公告](https://qh168.com.cn/article?article_id=304&cat_code=GongGao)：历史IC开平收费万分之0.23及每笔1元申报费资料。这里只作为成本情景依据，不将历史通知认作所有样本日期的完整费率审计。
- [中银期货收费公示](https://www.bocifco.com/newsinfo.aspx?cid=7&id=49925)：2026年公司标准与交易所费率不同。实际账户佣金未提供，基础加价及高费率敏感性均为研究假设。

网页资料查阅日期2026-10-27。公开合约规则不解决本地供应商是否调整价格、分钟标记与量单位的问题。旧research/只阅读README、配置结构与过程问题说明，没有复用代码、训练参数或权重。
