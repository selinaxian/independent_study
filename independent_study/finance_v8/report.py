import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iclab.data import dump_json,sha256
from finance_v8.train import ROOT,HERE,OUT

NAMES={'teacher':'新隐藏层＋事件教师STDP','autonomous':'新隐藏层＋自主输出STDP',
       'random':'随机隐藏层＋事件教师','shuffled':'新隐藏层＋错配事件教师',
       'fixed13':'新隐藏层固定1.3阈值','equal_hidden':'新隐藏层＋等权输出',
       'delay_ridge':'相同延迟输入监督Ridge','current_rv':'当前5分钟波动',
       'v7':'旧v7直接教师STDP','baseline':'原方向策略，无风险层'}


def main():
    cfg=json.loads((HERE/'config.json').read_text())
    metrics=pd.read_csv(OUT/'event_metrics.csv').set_index('model')
    summary=pd.read_csv(OUT/'trading_summary.csv')
    intervals=pd.read_csv(OUT/'event_paired_intervals.csv')
    leads=pd.read_csv(OUT/'event_summary.csv').set_index('model')
    neural=pd.read_csv(OUT/'neural_diagnostics.csv')
    paths=pd.read_csv(OUT/'held_path_metrics.csv')
    train=json.loads((OUT/'training_status.json').read_text())
    checks=json.loads((OUT/'prediction_verification.json').read_text())
    trades_check=json.loads((OUT/'trading_verification.json').read_text())
    assert trades_check['source_sha256']==sha256(HERE/'trade.py')
    base=summary[summary.scenario=='base'].set_index(['application','model'])
    frames=pd.read_parquet(OUT/'forecasts.parquet');quiet=frames[frames.covered]
    association=quiet.teacher.corr(quiet.equal_hidden)
    fold_association=quiet.groupby('fold').apply(lambda x:x.teacher.corr(x.equal_hidden),include_groups=False)
    dump_json(OUT/'signal_diagnostics.json',dict(teacher_equal_hidden_correlation=association,
               mean_within_fold_correlation=float(fold_association.mean()),
               low_information_warning='High correlation is diagnostic, not proof of identical information.'))
    event_table='\n'.join(f'|{NAMES[m]}|{metrics.loc[m,"auc"]:.4f}|{metrics.loc[m,"ap"]:.4f}|{metrics.loc[m,"precision"]:.2%}|{metrics.loc[m,"coverage"]:.2%}|' for m in cfg['models'])
    interval_table='\n'.join(f'|{NAMES[r.other]}|{r.metric}|{r.difference:.4f}|[{r.lower:.4f}, {r.upper:.4f}]|' for r in intervals.itertuples())
    def trade_table(application):
        rows=[]
        for model in ['baseline']+cfg['models']:
            r=base.loc[(application,model)]
            rows.append(f'|{NAMES[model]}|{int(r.trades)} / {int(r.lots)}|{r.net_pnl:,.2f}|{-r.max_drawdown_pct:.2f}%|{r.covered_selection_excess:,.2f}|[{r.covered_selection_excess_low:,.0f}, {r.covered_selection_excess_high:,.0f}]|')
        return '\n'.join(rows)
    stress='\n'.join(f'|{r.scenario}|{r.application}|{r.net_pnl:,.2f}|{-r.max_drawdown_pct:.2f}%|' for r in summary[summary.model=='teacher'].itertuples())
    path_table='\n'.join(f'|{NAMES[r.model]}|{r.target}|{r.pearson:.4f}|{r.spearman:.4f}|' for r in paths[(paths.scenario=='base')&(paths.scope=='covered')&paths.model.isin(['teacher','autonomous','current_rv'])].itertuples())
    example=pd.read_csv(OUT/'backtests/base__sizing__teacher_opportunities.csv')
    examples=[]
    for quantity in [2,1,0]:
        choices=example[example.covered & (example.lots==quantity)]
        if choices.empty:continue
        r=choices.iloc[0];rule=json.loads((OUT/'folds'/f'fold_{int(r.fold):02d}'/'lock.json').read_text())['rules']['teacher']
        examples.append(f'|{r.signal_time}|{r.contract_id}|{"多" if r.side>0 else "空"}|{r.risk_score:.4f}|{rule["q60"]:.4f} / {rule["q80"]:.4f}|{quantity}手|')
    first_table='\n'.join(examples)
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    order=cfg['models'];colors=['#dd713b' if m=='teacher' else '#3c83a4' if m in ['current_rv','delay_ridge'] else '#9ca8b0' for m in order]
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,key,reference in zip(axes,['auc','ap'],[.5,float(metrics.loc['teacher','base_rate'])]):
        ax.barh(order,[metrics.loc[m,key] for m in order],color=colors)
        ax.axvline(reference,color='black',linestyle='--',linewidth=1)
        ax.invert_yaxis();ax.set_title('Event '+key.upper());ax.set_xlim(0,.8 if key=='auc' else .23)
        ax.grid(axis='x',alpha=.2)
    fig.savefig(figures/'event_prediction.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4.6),layout='constrained')
    for model,label,color in [('baseline','Direction only','#858585'),('teacher','Teacher STDP filter','#dc713b'),
                              ('autonomous','Autonomous STDP filter','#8363a6'),('current_rv','Current RV filter','#3388a6')]:
        daily=pd.read_csv(OUT/f'backtests/base__filter__{model}_daily.csv')
        ax.plot(pd.to_datetime(daily.trading_date),daily.equity,label=label,color=color)
    ax.set_title('Conditional equity: 1-lot risk filters, 1 tick per side')
    ax.set_ylabel('Equity (CNY)');ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.savefig(figures/'filter_equity.png',dpi=160);plt.close(fig)
    tm=metrics.loc['teacher'];b=base.loc[('filter','baseline')];f=base.loc[('filter','teacher')]
    sz=base.loc[('sizing','teacher')];sb=base.loc[('sizing','baseline')];timing=base.loc[('timing','teacher')]
    text=f'''# v8金融信号与交易应用报告

已完成“真实分钟特征→延迟隐藏层→事件/自主STDP输出→风险信号→条件交易”的全流程。**新结构在本轮真实金融数据上没有建立可靠事件预测增量，90组固定交易评价没有净盈利。** 过滤或缩仓减少亏损，但同覆盖范围、同暴露的随机控制没有确认主模型的选时优势。不能把之前合成序列成功当成金融有效。

主模型事件AUC {tm.auc:.4f}、AP {tm.ap:.4f}，当前波动基线AUC {metrics.loc['current_rv','auc']:.4f}。1手过滤净亏 {abs(f.net_pnl):,.2f}元，无风险层净亏 {abs(b.net_pnl):,.2f}元；少亏 {f.net_change:,.2f}元，但对应配对区间[{f.net_change_low:,.0f}, {f.net_change_high:,.0f}]跨零。该结果仍不构成可采用的盈利策略。

## 1. 本次实际完成了什么

- 23段40日训练/10日验证/5包日期预测，2026-04-01—09-15，共{train['forecast_minutes']}个有效分钟，其中{train['quiet_decisions']}个静默事件评价时点，与v7严格对齐。
- 69个训练隐藏层、69个冻结随机隐藏对照、345组输出STDP权重、23个同延迟监督Ridge，训练约{train['seconds']:.1f}秒。每折重新初始化，预测时冻结。
- 9风险模型/对照×3用途×3执行情景，加各用途基准，共90组条件回测；固定672个原方向机会，60480条逐机会记录独立核账。
- 事件预测、独立事件提前量、交易持有期波动/不利路径、同暴露及同覆盖随机控制、成本/延迟压力、报告与原始产物已保存。动态止损及盘口挂单没有实施。

## 2. 数据、特征、编码与结构

只读原真实合约底表43440分钟，字段含上海时区datetime、contract_id、OHLC、volume、open_interest、segment_id和来源标记；沿用原前一包日OI合约选择及12项因果特征。具体为1/3分钟变化相对ATR、5分钟路径效率、支撑/阻力距离、收盘位置、短长波动比、振幅比、成交量异常、持仓变化、波动水平和已知日内时钟。它们是候选特征，不预先假定预测性。

每项特征训练20/80分位的低/高穿越形成两路脉冲，共24路。每路保留过去0—4分钟的延迟支路，共120个二值输入位置；跨日、午休、合约或缺分钟清空。隐藏层32个窗口独立IF组合单元，每个120条输入突触，输入→隐藏共3840条连接；隐藏→输出32条连接。

```mermaid
flowchart LR
    A[12个因果特征] --> B[24路阈值穿越脉冲]
    B --> C[0至4分钟延迟\n120个输入位置]
    C --> D[32个IF组合单元]
    D --> E[32条STDP输出连接]
    E --> F[连续风险分数]
    F --> G[过去验证分位门槛]
    G --> H[过滤 / 分档手数 / 预警择时]
    I[原始特征方向策略] --> H
    J[训练中已发生价格事件] -. 教师版本 .-> E
```

隐藏训练复用竞争诱发post的STDP增强和入权重总量约束变体，不读取事件标签；不是自由运行LIF成对STDP。推理单窗积分、阈值发放并重置，时间记忆由延迟支路提供。

真实数据每个5分钟输入窗平均有{neural.train_input_spikes.mean():.2f}个脉冲，远比合成的2个密集且分散。主版本每个隐藏阈值取训练驱动90%分位，并严格大于单连接最大权重；平均阈值{neural.theta_mean.mean():.4f}，训练平均发放率{neural.train_rate.mean():.2%}、预测{neural.forecast_rate.mean():.2%}。随机对照也按自身训练驱动校准活动量，没有利用事件标签或预测结果设阈值。

原合成固定阈值1.3完整保留为对照，真实预测发放率仅{neural.fixed_forecast_rate.mean():.6%}，几乎沉默；它只有极少告警，不能因为合并AUC略高于0.5就视为有用信号。其跨折零值/标尺细微差异也可能影响合并排序。主版本每折/种子预测均32个单元活跃，平均权重余弦{neural.mean_cosine.mean():.4f}；活跃和多样性本身不能证明金融预测有效。

## 3. 输出到底是什么，是否有监督

主风险分数为三种子平均：`S(t)=mean_seed(sum_j output_weight[j] × hidden_spike[j,t])`。它是一个加权响应数值，没有涨跌方向，不是事件发生概率。

教师事件仍为绝对1分钟对数收益超过训练95%分位且此前5分钟无突破。教师版只用训练历史的实际事件当post，隐藏脉冲当pre，按1—5分钟先后增减输出权重；隐藏先学组合，教师再给组合赋事件权重。这是事件教师驱动的混合学习，价格事件没有反传更新隐藏连接。

本轮同时完成autonomous对照：输出IF根据当前加权隐藏输入自主发放，STDP配对使用近期内部脉冲，输出阈值只按无标签活动量适配。该路不读取价格事件训练权重，但是否对应风险仍须用事件事后评价。没有偷偷将它称作教师版或宣称自主输出天然代表价格事件。

其他对照包括随机隐藏、错配事件教师、等权输出、固定1.3、同120维延迟输入监督Ridge、当前5分钟波动，以及复载原v7。所有模型使用同一目标、日期和评价样本。

主模型分数与等权隐藏发放分数的整体相关为{association:.4f}，各折平均{fold_association.mean():.4f}；教师权重均值{neural.teacher_w_mean.mean():.4f}、神经元间标准差均值{neural.teacher_w_std.mean():.4f}。这提示当前教师加权后排序仍主要跟随隐藏活动量，不足以证明两者信息完全相同。

## 4. 真实事件预测与提前量

当前最近5分钟无阈值突破时，预测未来1—5分钟事件；实际正例比例{tm.base_rate:.2%}。分数用过去验证分布的经验中秩分位统一展示，恒定分数不会被当成概率1；告警按原始分数严格大于过去验证80%分位。

|模型|AUC|AP|告警精确率|告警覆盖率|
|---|---:|---:|---:|---:|
{event_table}

![事件预测对照](figures/event_prediction.png)

AUC随机约0.5，AP参照10.62%的正例比例。主模型没有超过自主/等权/错配输出，弱于同延迟监督和当前波动；相对旧v7没有确认改善。主模型在{int(leads.loc['teacher','evaluable_events'])}个可评价事件中命中{int(leads.loc['teacher','detected_events'])}个，覆盖{leads.loc['teacher','event_recall']:.2%}；已命中事件最早提前中位{leads.loc['teacher','median_earliest_lead']:.0f}分钟，不能忽略漏报而只展示提前量。完整537个事件含因预热等原因没有有效提前决策点的记录。

5日块1000次配对95%区间（教师版减对照，探索性，未作多重比较校正）：

|对照|指标|差值|区间|
|---|---|---:|---|
{interval_table}

各折事件频率明显漂移，部分折无正例，相关指标保留NaN；分折/月与种子结果均在CSV中，不根据本轮结果倒转分数或改选阈值。

## 5. 信号怎样变成交易

多空来自**已有v2全滚动原始特征收益预测**：预测正且超过原验证门槛做多，预测负且超过门槛做空。新STDP只改变是否接纳该机会及手数，没有提供新的方向预测。

保持原672个非重叠方向机会，被跳过的机会仍占用原虚拟5分钟时钟，不借此额外抢开下一笔。本次486个机会处于新事件目标适用的静默范围，另186个机会保持过滤/仓位基准，不用超出适用范围的分数调仓；高风险择时中无适用预警就不开仓。所有风险模型共享这一覆盖范围。它与v4对所有机会用RV缩仓的覆盖方式不同，不能直接拿v4盈亏判断本版退步幅度。

|用途|规则|本金与基准|
|---|---|---|
|低风险过滤|分数超过过去验证80%分位，跳过；否则原方向1手|100万，基准每机会1手|
|分档仓位|≤60%分位2手，60—80%之间1手，>80%分位0手|200万，基准每机会2手|
|高风险择时|仅有高风险预警时按原方向1手，其余0手|100万，基准每机会1手|

这是相对风险档位，不是把STDP分数直接当波动率去反比例分仓。没有事件方向时，风险高既可能意味着持仓获利也可能意味着亏损，因此三用途在运行前分别固定，保留全部结果。

以下为各仓位档第一次真实出现的机会（只按时间和当时风险选择例子，不按事后盈亏挑选）：

|信号时间|实际合约|原方向|STDP分数|过去60% / 80%门槛|分档仓位|
|---|---|---|---:|---|---|
{first_table}

信号在t收盘形成，基准按t+2的分钟bar open进场、t+7的open平仓，持有5分钟；按照暂用的分钟结束标记解释，信号后至少留出一完整分钟。无跨午休/隔夜/合约持仓。IC乘数200元/点、tick0.2点，单边1tick滑点；沿用研究费用假设：开仓0.000023、平今0.00023、另加经纪费率0.00001及每单1元。费用率是本项目假设，不是重新核实的账户实际费率。

原事件窗口t+1…t+5与交易持有区间不完全重合，因此还单独评价实际持有期风险，不能把预警命中率直接当止损收益。

## 6. 回测结果

表中“严格同暴露增量”将手数只在同折、同静默覆盖状态内随机置换，保持非适用时段不变；它比仅保持总交易量更能分离风险分数的选择作用。原同折总体置换也保留，补充过程见[归因核验说明](../../finance_v8/EVALUATION_NOTES.md)。

### 低风险过滤：100万，原1手

|模型|交易笔数 / 总手数|净盈亏（元）|最大回撤|严格同暴露增量（元）|95%区间|
|---|---:|---:|---:|---:|---|
{trade_table('filter')}

![过滤策略净值](figures/filter_equity.png)

### 分档仓位：200万，原2手

|模型|交易笔数 / 总手数|净盈亏（元）|最大回撤|严格同暴露增量（元）|95%区间|
|---|---:|---:|---:|---:|---|
{trade_table('sizing')}

### 仅高风险时择时：100万，1手

|模型|交易笔数 / 总手数|净盈亏（元）|最大回撤|严格同暴露增量（元）|95%区间|
|---|---:|---:|---:|---:|---|
{trade_table('timing')}

主教师过滤少亏{f.net_change:,.2f}元，但严格同暴露增量为{f.covered_selection_excess:,.2f}元，区间[{f.covered_selection_excess_low:,.0f}, {f.covered_selection_excess_high:,.0f}]。分档仓位净亏{abs(sz.net_pnl):,.2f}元，相应2手基准净亏{abs(sb.net_pnl):,.2f}元；两本金的绝对亏损不可混比。高风险择时只做{int(timing.trades)}笔，仍净亏{abs(timing.net_pnl):,.2f}元；回撤较小主要伴随大幅减少暴露，不能当预警已有效。

原1手基准毛收益仅{b.gross_pnl:,.2f}元，手续费{b.fees:,.2f}元、滑点{b.slippage:,.2f}元；新教师过滤后毛收益{f.gross_pnl:,.2f}元。因此当前既有方向机会的毛利不足覆盖成本，风险层也未改善机会选择到足以盈利，不能只把失败归于某一项手续费。

全部90组无净盈利；固定1.3的高风险择时因无交易为0，不是有效盈利。其他对照即使某项相对随机区间为正，也不代表净盈利或经独立未见数据确认；完整逐项差异保留，不事后换成最佳模型。

## 7. 成本/延迟与持有路径核对

主教师模型压力结果：

|执行情景|用途|净盈亏（元）|最大回撤|
|---|---|---:|---:|
{stress}

extra_minute把入出场同时延后1分钟，仍持有5分钟，所有机会通过同一连续时段核验。风险分数/门槛不随成本情景重选。

486个适用交易机会的实际持有区间相关：

|模型|事后目标|Pearson|Spearman|
|---|---|---:|---:|
{path_table}

主教师分数未呈现有利的持有期波动或方向条件不利路径关系；当前波动与后续实现波动关系更强，但高波动与交易亏损不是同一概念。没有把此分数直接用作动态止损距离。

## 8. 验证、结论和文件

6项新接入/归因测试与19项复用机制/会计测试共25项通过。23折预处理、标尺、日期及源哈希核对；首/中/末3折×3种子的模型从原始窗口复载，验证/预测共{checks['reconstructed_split_model_outputs']}组模型输出最大差{checks['maximum_prediction_error']:.3g}。全部v7样本/标签/原始分数一致，{checks['old_artifacts_unchanged']}个v7/v8合成旧产物未变。

60480条逐机会独立按成交价、方向、手数、费用与滑点重算，最大差{trades_check['maximum_accounting_error']:.3g}元；基准净值精确重现已有v2，所有情景保证金拒单为{trades_check['maximum_margin_rejections']}。常数风险分数的相关系数显式保留NaN，不以0或虚假的好成绩替代。

这条新链路已经完整运行，但还不能作为已验证的交易信号。当前证据更支持继续定位“隐藏表示与事件目标的联系”：输入延迟的监督基线仍保留一些信息，教师输出与等权活动量高度相似，合成模式能力没有转化为金融预警优势。未来若改负反馈、目标或编码，应另立实验并寻找新留出数据，不能继续在本历史上调参后称独立验证。

本轮全为已观察历史的探索性滚动/条件执行模拟。真实分钟时间标签、价格调整和量单位仍未获供应商确认，执行使用bar open和假设成本，不能称已验证实盘。报告不回避负结果，也不将暂时少亏等同于盈利能力。

复现见[README](../../finance_v8/README.md)，运行前规则见[协议](../../finance_v8/PROTOCOL.md)。关键产物：[事件指标](event_metrics.csv)、[事件差异区间](event_paired_intervals.csv)、[全部交易汇总](trading_summary.csv)、[同模型配对收益](paired_trading_intervals.csv)、[逐月结果](trading_monthly.csv)、[持有期风险关系](held_path_metrics.csv)、[逐事件提前量](event_leads.csv)。folds/保存权重/标尺/验证预测，backtests/保存逐机会和资金曲线。
'''
    (OUT/'report.md').write_text(text,encoding='utf-8')
    lock=json.loads((OUT/'source_lock.json').read_text())
    for kind in ['sources','old_artifacts']:
        for path,expected in lock[kind].items():assert sha256(ROOT/path)==expected,path
    verification={**checks,**trades_check,'tests_passed':25,'training_complete':True,'backtests_complete':True}
    dump_json(OUT/'verification.json',verification)
    dump_json(ROOT/'outputs/independent_v8_finance_verification.json',verification)
    artifacts={str(p.relative_to(ROOT)):sha256(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}
    sources={str(p.relative_to(ROOT)):sha256(p) for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    dump_json(OUT/'manifest.json',dict(status='complete',artifacts=artifacts,sources=sources))
    print('Report complete:',OUT/'report.md', 'artifacts',len(artifacts),flush=True)


if __name__=='__main__':main()
