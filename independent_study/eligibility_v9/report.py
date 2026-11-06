import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iclab.data import dump_json,sha256
from finance_v8.controls import safe_corr
from eligibility_v9.train import ROOT,HERE,OUT,ARMS

NAMES={'unsupervised':'第一层无反馈（原v8）','true_feedback':'第一层真实事件反馈',
       'mismatched':'第一层错配事件反馈','current_rv':'当前5分钟波动',
       'delay_ridge':'同120维延迟输入Ridge','baseline':'原方向策略、不加风险控制'}


def table(frame,cols):
    def cell(x):
        if isinstance(x,(float,np.floating)):return f'{x:.6f}'
        return str(x)
    return '|'+ '|'.join(cols)+'|\n|'+'|'.join(['---']*len(cols))+'|\n'+'\n'.join(
        '|'+ '|'.join(cell(row[c]) for c in cols)+'|' for _,row in frame.iterrows())


def main():
    cfg=json.loads((HERE/'config.json').read_text())
    train=json.loads((OUT/'training_status.json').read_text())
    checks=json.loads((OUT/'prediction_verification.json').read_text())
    tradecheck=json.loads((OUT/'trading_verification.json').read_text())
    metrics=pd.read_csv(OUT/'event_metrics.csv').set_index('model')
    interval=pd.read_csv(OUT/'event_paired_intervals.csv')
    trades=pd.read_csv(OUT/'trading_summary.csv');base=trades[trades.scenario=='base'].set_index(['application','model'])
    paired=pd.read_csv(OUT/'paired_trading_intervals.csv')
    leads=pd.read_csv(OUT/'event_summary.csv')
    neural=pd.read_csv(OUT/'neural_diagnostics.csv');feedback=pd.read_csv(OUT/'feedback_audit.csv')
    probe=pd.read_csv(OUT/'hidden_probe_metrics.csv');units=pd.read_csv(OUT/'unit_associations.csv')
    paths=pd.read_csv(OUT/'held_path_metrics.csv')
    mean=neural.groupby('model').mean(numeric_only=True)
    signal_rows=[]
    forecasts=pd.read_parquet(OUT/'forecasts.parquet')
    bars=pd.read_parquet(ROOT/'independent_study/results/v1/bars_features.parquet')
    from iclab.data import FEATURES
    from event_stdp_v7.core import encode
    from finance_v8.core import delayed_history
    for fid,g in forecasts.groupby('fold'):
        fd=OUT/'folds'/f'fold_{fid:02d}';p=np.load(fd/'preprocess.npz')
        s=encode(bars[FEATURES].to_numpy(),bars.segment_id.to_numpy(),p['bounds'])
        x,valid=delayed_history(s,bars[FEATURES].to_numpy(),bars.segment_id.to_numpy(),cfg['delays'])
        idx=g.loc[g.covered,'bar_id'].to_numpy()
        for arm in ARMS:
            counts=[];weights=[]
            for seed in cfg['seeds']:
                m=np.load(fd/f'{arm}_{seed}.npz');h=(x[idx]@m['hidden'].T>m['theta']).astype(float)
                counts.append(h.sum(axis=1));weights.append(m['output'])
            signal_rows.append(dict(fold=fid,model=arm,
                score_activity_correlation=safe_corr(g.loc[g.covered,arm].to_numpy(),np.mean(counts,axis=0)),
                fraction_output_above_initial=float((np.concatenate(weights)>.5).mean())))
    signal=pd.DataFrame(signal_rows);signal.to_csv(OUT/'signal_diagnostics.csv',index=False)
    signal_mean=signal.groupby('model')[['score_activity_correlation','fraction_output_above_initial']].mean().reset_index()
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    for ax,key,ref in zip(axes,['auc','ap'],[.5,float(metrics.loc['true_feedback','base_rate'])]):
        ax.barh(cfg['models'],[metrics.loc[m,key] for m in cfg['models']],color=['#999999','#dc713b','#8964ac','#3388a6','#5a9f84'])
        ax.axvline(ref,color='black',ls='--',lw=1);ax.invert_yaxis();ax.set_title('Event '+key.upper());ax.grid(axis='x',alpha=.2)
    fig.savefig(figures/'event_prediction.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4.3),layout='constrained')
    for m in ['baseline']+ARMS+['current_rv']:
        d=pd.read_csv(OUT/f'backtests/base__filter__{m}_daily.csv')
        ax.plot(pd.to_datetime(d.trading_date),d.equity,label=m)
    ax.set_title('Same direction opportunities: 1-lot filters, 1 tick per side')
    ax.set_ylabel('Equity (CNY)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(figures/'filter_equity.png',dpi=160);plt.close(fig)
    et='\n'.join(f'|{NAMES[m]}|{metrics.loc[m,"auc"]:.4f}|{metrics.loc[m,"ap"]:.4f}|{metrics.loc[m,"precision"]:.2%}|{metrics.loc[m,"coverage"]:.2%}|' for m in cfg['models'])
    ci='\n'.join(f'|{NAMES[r.other]}|{r.metric}|{r.difference:.5f}|[{r.lower:.5f}, {r.upper:.5f}]|' for r in interval.itertuples())
    nt='\n'.join(f'|{NAMES[m]}|{mean.loc[m,"train_rate"]:.3%}|{mean.loc[m,"forecast_rate"]:.3%}|{mean.loc[m,"active_forecast"]:.2f}|{mean.loc[m,"cosine"]:.4f}|{mean.loc[m,"hidden_l1_change"]:.5f}|' for m in ARMS)
    def trade_table(app):
        out=[]
        for m in ['baseline']+cfg['models']:
            r=base.loc[(app,m)]
            out.append(f'|{NAMES[m]}|{int(r.trades)}/{int(r.lots)}|{r.net_pnl:,.2f}|{-r.max_drawdown_pct:.2f}%|{r.covered_selection_excess:,.2f}|[{r.covered_selection_excess_low:,.0f}, {r.covered_selection_excess_high:,.0f}]|')
        return '\n'.join(out)
    stress='\n'.join(f'|{r.scenario}|{r.application}|{r.net_pnl:,.2f}|{-r.max_drawdown_pct:.2f}%|' for r in trades[trades.model=='true_feedback'].itertuples())
    selected=units[units.selected].copy();selected['train_lift']=selected.train_event_probability-selected.train_base
    selected['forecast_lift']=selected.forecast_event_probability-selected.forecast_base
    sel=selected.groupby('model')[['train_lift','forecast_lift']].mean().reset_index()
    summary_diags=mean[['positive_delta','negative_delta','unsupervised_delta','positive_updates','negative_updates']].reset_index()
    pos_ratio=mean.loc['true_feedback','positive_delta']/mean.loc['true_feedback','unsupervised_delta']
    neg_ratio=mean.loc['true_feedback','negative_delta']/mean.loc['true_feedback','unsupervised_delta']
    trade_pairs=paired[(paired.scenario=='base')&paired.other.isin(['unsupervised','mismatched'])]
    held=paths[(paths.scenario=='base')&(paths.scope=='covered')&(paths.target=='held_rv')]
    main=metrics.loc['true_feedback'];fil=base.loc[('filter','true_feedback')]
    d=interval[(interval.other=='unsupervised')&(interval.metric=='auc')].iloc[0]
    ms=interval[(interval.other=='mismatched')&(interval.metric=='auc')].iloc[0]
    conclusion=('本次AUC增量区间同时高于无反馈和错配组，需要结合AP及经济评价判断用途。' if d.lower>0 and ms.lower>0 else
                '本轮尚未确认真实事件反馈相对无反馈、错配反馈的稳定预测增量。')
    profitable=int((trades.net_pnl>1e-6).sum())
    content=f'''# v9：第一层事件调制资格迹，完整受控实验报告

23折、3种子、3组共{train['hidden_models']}个隐藏模型和{train['output_models']}个输出模型已完成。**{conclusion}** 主模型事件AUC {main.auc:.4f}、AP {main.ap:.4f}；真实反馈过滤净收益 {fil.net_pnl:,.2f}元。共{tradecheck['cases']}组预设条件回测，净盈利组数{profitable}。结果不等于可部署策略。

## 1. 改了什么，保持了什么

仅第一层增加事件调制项；无反馈、真实反馈、错配反馈三组使用相同初始化、训练样本、随机顺序、3轮、延迟、32个隐藏单元、隐藏阈值规则和真实事件输出教师。第二层绝没有在错配组换成错配教师。无反馈69组权重/阈值/输出及23折预测精确重现v8，9个执行/应用组合的逐笔仓位及盈亏也重现v8教师模型。因而此次不是同时改网络、读出与策略。

每折40训练日、10验证日、5预测包日期，重置后重新拟合，不是权重持续接续的在线训练。2026-04-01—09-15共{train['forecast_minutes']}个有效预测分钟、{train['quiet_decisions']}个静默评价时点。三组完全共享日期、标的和目标；其中106个实际行情日。当前RV与同输入Ridge直接复用v8冻结结果。

## 2. 数据、特征、编码和结构

沿用标准真实IC合约底表选出的181日43440行：datetime（上海时区）、trading_date、contract_id、OHLC、volume、open_interest、bar_id、segment_id、来源审计字段。每天按此前包日期OI选择当日可用合约；合约/半日/缺口分段，特征不跨段混算。输入是已有因果特征表`results/v1/bars_features.parquet`，并非以主连价格当成交价。

12特征为：1/3分钟价格变化除以前20分钟ATR；5分钟方向效率；距此前20分钟最高/最低价的阻力/支撑距离除ATR；当根收盘在高低区间的位置；短长波动比；当前振幅/历史ATR；成交量相对历史均量的异常；log持仓变化；历史波动水平；已知交易时钟。前20分钟统计均不包含未来。指标只是候选经济信息，支撑阻力不天然保证预测性；本轮未添加新因子。

各特征训练20/80分位低/高阈值穿越产生2路事件脉冲，共24路；持续高值不重复放电。每路扩展0、1、2、3、4真实分钟延迟，得到120维到达输入。120→32竞争IF组合单元→1个加权隐藏脉冲风险分数→验证80分位告警。窗口独立积分/阈值/重置，历史由延迟输入保存；不是自由运行的多层LIF连续膜电位网络。

第一层训练通过竞争诱发胜者post，增强活动pre连接并约束入权重总量；训练诱发post不要求先跨过预测阈值。预测隐藏阈值=max(训练驱动90分位, 最大单突触权重+1e-6)。输出层保持1—5真实分钟pre/post时序STDP：隐藏在事件前则增强，事件后响应则减弱；输出为Σ隐藏脉冲×输出权重，不是概率、涨跌方向或自动止损价。

## 3. 资格迹反馈的准确含义

在历史训练样本t，胜者j的参与连接获得 e_ij=pre_i×post_j。完整5分钟观察窗结束后，e_ij(t+5)=exp(-5/5)e_ij(t)。y_t表示(t,t+5]内是否出现价格事件，M=y_t-p_train。事件仍为绝对1分钟收益超过训练95分位、此前5分钟未突破的首次事件。

原无监督增强 Δu=0.15(1-w)pre 保留。新增 Δm=0.15 M e b(w,M)：M≥0时b=1-w，M<0时b=w；两项基于更新前w，然后沿用总量2归一化及[0,1]裁剪。正反馈强化参与连接，负反馈给参与连接一个负辅助项。由于Δu和归一化并存，不能说“无事件后每个最终权重一定下降”。中心化M也不使有界权重增量严格均值为零。

本版是**已成熟样本的离线资格迹重放**：完整历史训练期内重建局部pre/post资格，重放5分钟后的目标反馈；保持v8随机样本顺序以控制训练方式差异。没有跨样本持续累积的资格矩阵，也未按具体事件的第1/第5分钟给予不同信用。资格衰减只保留这一个固定5分钟龄期。验证/预测标签不进入第一层或第二层训练，预测冻结，不修改过去信号。

错配仅在训练半日内移动已成熟y标签，保留该段正例数和反馈适用范围；平均错配标签比例{feedback.mismatch_fraction.mean():.2%}。训练p范围[{feedback.p_train.min():.4f},{feedback.p_train.max():.4f}]。这是一种错位负控制，不是足够多次随机化的显著性检验。

## 4. 机制核验和合成实验中的限制

30项相关测试通过：局部连接归属、正负反馈符号、指数衰减、不跨训练缺口/时段、严格未来目标、错配计数、无反馈及零调制精确等价，以及复用的时序/执行规则测试。

合成8模板的训练90分位恰好等于最大驱动，严格“大于”使32/32单元沉默，三组AUC均0.5。这个失败完整保留于[v9合成结果](../v9_mechanism/metrics.csv)。附属事后解释只沿用旧v8已有固定1.3阈值、保留隐藏权重并重训同规则输出；三组3种子AUC/AP均1，见[阈值审计](../v9_mechanism/threshold_audit/metrics.csv)。它说明这些简单模板仍可读出，不说明真实反馈有独有优势；没有据此改变金融阈值或挑选金融赢家。

## 5. 第一层有没有改变，有没有更有预测性

|第一层|训练发放率|预测发放率|平均活跃单元/32|平均突触余弦|对无反馈平均绝对权重差|
|---|---|---|---|---|---|
{nt}

真实/错配反馈确实改变第一层权重与竞争路径。单元编号可能发生分工变化，权重差不是模式质量指标。原始无监督项和正负辅助项累计绝对量如下（各模型平均，滚动训练数据重叠，不能当独立样本数）：

{table(summary_diags,['model','positive_delta','negative_delta','unsupervised_delta','positive_updates','negative_updates'])}

真实反馈正/负辅助项的累计绝对量，分别约为原无监督增强的{pos_ratio:.2%}/{neg_ratio:.2%}。这是归一化之前的更新量，不等于最终权重贡献比例，但说明这次是较弱的辅助目标，主体仍由无监督竞争驱动；它检验不了强反馈或完全替换学习目标的效果。金融结果出来后没有重新搜索反馈力度。

每折/种子只在训练期选事件条件概率最高且至少20次响应的隐藏单元，预测期不重新挑选。其预测结果如下，response为3种子所选单元响应的平均；这是标签辅助诊断，不是主模型新读出或交易策略。

{table(probe,['model','auc','ap','precision','coverage'])}

训练选择单元的平均事件概率增量（条件概率减该折总体事件率）如下。跨折等权均值仅作描述，不能代替独立样本检验或挑选预测期表现好的神经元：

{table(sel,['model','train_lift','forecast_lift'])}

本轮训练选择单元的集成AUC：真实反馈{probe.set_index('model').loc['true_feedback','auc']:.4f}，无反馈{probe.set_index('model').loc['unsupervised','auc']:.4f}，错配{probe.set_index('model').loc['mismatched','auc']:.4f}，没有呈现清晰优势。少量单元仍可能含有未被此简单诊断提取的信息，不能据此证明全部信息丢失。

全部32单元、所有折/种子详见[unit_associations.csv](unit_associations.csv)。隐层结构改变、发放正常都不能自动证明学到预测性组合。

最终分数与隐藏层总活动量的逐折平均相关，以及输出连接超过初值0.5的比例：

{table(signal_mean,['model','score_activity_correlation','fraction_output_above_initial'])}

高相关提示输出仍很大程度反映总活动量，未证明选择性组合信息完全消失。本轮保持输出层不变，不能把现有输出限制的可能性排除。

## 6. 最终事件风险分数

总体未来事件比例{main.base_rate:.2%}。每折分数先按过去验证分布转换分位，避免直接混合不同尺度；告警仍为原始分数超过过去验证80分位。

|模型|AUC|AP|告警精确率|告警覆盖率|
|---|---|---|---|---|
{et}

真实反馈减其他模型的配对5日块95%区间（1000次；未因多组比较调整，均为探索性）：

|参照|指标|差值|95%区间|
|---|---|---|---|
{ci}

这些区间以本次已拟合模型为条件，按日期块重采样预测结果，没有在每次bootstrap内重训网络。真实反馈的AUC微升，相对无反馈/错配均跨零，AP也未确认增量；当前波动与同输入Ridge仍明显更强。第一层标签接入已经实现，是否形成可泛化的组合仍未通过。

![事件预测](figures/event_prediction.png)

{table(leads,['model','total_events','evaluable_events','detected_events','event_recall','median_earliest_lead'])}

提前量中位数只针对被检测事件，不能忽略漏报；537个全部事件与405个具有此前可评价静默机会的事件均保留。AUC/AP评价的是事件发生前的排序，并非对“金融因果”的证明。按月/折/种子结果见同目录CSV。

## 7. 如何转成交易信号

方向继续来自v2原始特征收益预测，沿用672个固定交易机会；风险模型只改变是否交易/几手。486个静默适用机会可用新风险评分，其余186机会过滤和分档保持原仓位，高风险择时不开仓。跳过机会仍保留原虚拟持仓时钟，不补开新机会。

- 过滤：风险>验证80分位则0手，其余1手；本金100万元。
- 分档：风险≤60分位2手，60—80分位1手，>80分位0手；本金200万元。
- 高风险择时：只在>80分位时按原方向1手；本金100万元。这是对照用途，不预设高波动有利。

t时刻收盘可知信号，t+2分钟bar open入、持有5分钟在t+7 open出。同合约/同半日；不跨午休、隔夜或换月。乘数200、tick0.2；开仓费率0.000023、平今0.00023、经纪额外0.00001、每订单1元，单边1tick。另跑2tick和额外延迟1分钟。没有新增动态止损或挂单预测。

### 1手过滤（100万本金）

|模型|笔/手|净收益元|最大回撤|同覆盖/同暴露选择增量元|增量95%区间|
|---|---|---|---|---|---|
{trade_table('filter')}

### 2/1/0手分档（200万本金）

|模型|笔/手|净收益元|最大回撤|同覆盖/同暴露选择增量元|增量95%区间|
|---|---|---|---|---|---|
{trade_table('sizing')}

### 仅高风险择时（100万本金）

|模型|笔/手|净收益元|最大回撤|同覆盖/同暴露选择增量元|增量95%区间|
|---|---|---|---|---|---|
{trade_table('timing')}

同暴露控制在相同fold且相同静默适用范围内置换仓位，保留交易数量/手数，不把简单少交易的少亏当选时价值。原全fold控制仍保留CSV；上表严格控制为主。不同本金的绝对盈亏不可直接比较。

真实反馈相对三组及基线的逐日配对交易区间见[paired_trading_intervals.csv](paired_trading_intervals.csv)。

{table(trade_pairs,['application','other','true_feedback_minus_other','lower','upper'])}

真实反馈过滤相对无反馈少亏{base.loc[('filter','true_feedback'),'net_pnl']-base.loc[('filter','unsupervised'),'net_pnl']:,.2f}元，但对应区间跨零；同覆盖/同暴露选择增量也跨零。分档同样未确认收益增量，而仅高风险交易反而比原模型多亏。不能将其中一个较好的历史账面数称为已验证改进。

![过滤净值](figures/filter_equity.png)

|真实反馈执行情景|用途|净收益元|最大回撤|
|---|---|---|---|
{stress}

事件预测窗口为未来1—5分钟，实际执行为2—7分钟，两者不同；[held_path_metrics.csv](held_path_metrics.csv)单独检验持有期实现波动、绝对收益与方向不利路径。不把事件AUC当止损收益证据。

{table(held,['model','n','pearson','spearman'])}

上表为信号与实际持有期波动的关系，真实反馈仍接近零；当前波动基线保留明显更强的风险信息。风险预测尚不适合被包装为可用动态止损信号。

## 8. 核验、解释边界与交付

所有23折特征阈值、事件阈值、反馈成熟边界、验证分位标尺和文件锁均核验；首/中/末3折×3组×验证/预测共{checks['reconstructed_split_model_outputs']}组集成预测复载误差{checks['maximum_prediction_error']}，另{checks['hidden_retrained_exact']}个隐藏模型从头重训精确一致。无反馈69模型与v8完全相同。

{tradecheck['opportunity_records']}条逐机会记录按价格、方向、手数独立核账，最大差{tradecheck['maximum_accounting_error']:.3g}元；保证金拒单{tradecheck['maximum_margin_rejections']}。原v8的{checks['old_artifacts_unchanged']}个锁定文件未改。机制、权重、逐分钟预测、逐机会交易、区间、图和配置均可复查；入口见[README](../../eligibility_v9/README.md)、运行前[协议](../../eligibility_v9/PROTOCOL.md)。

本轮回答的是“这一预定的局部资格迹辅助项，能否在原结构上带来增量”。即便未改善，也不能断言所有SNN、所有STDP或所有事件调制都无效；这里保持了原竞争机制、分布式组合、阈值、输出汇总和有限反馈强度，仍存在表示能力与读出限制。反之，单次少亏也不够证明预测目标成功进入后产生稳定经济价值。

全部日期此前已观察，本轮为探索性滚动历史研究。分钟时间标签、供应商价格调整及量单位未完全确认，回测使用bar open和假设成本，只是条件执行模拟。当前没有实盘盈利或新留出样本的验证结论。
'''
    (OUT/'report.md').write_text(content)
    for key in ['sources','old_artifacts']:
        for path,expected in json.loads((OUT/'source_lock.json').read_text())[key].items():assert sha256(ROOT/path)==expected,path
    verification=dict(status='complete',training=train,prediction=checks,trading=tradecheck,related_tests_passed=30,
        report=str((OUT/'report.md').relative_to(ROOT)))
    dump_json(OUT/'verification.json',verification)
    dump_json(ROOT/'outputs/independent_v9_eligibility_verification.json',verification)
    artifacts={str(p.relative_to(ROOT)):sha256(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}
    for p in (ROOT/'independent_study/results/v9_mechanism').rglob('*'):
        if p.is_file():artifacts[str(p.relative_to(ROOT))]=sha256(p)
    sources={str(p.relative_to(ROOT)):sha256(p) for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    dump_json(OUT/'manifest.json',dict(status='complete',artifacts=artifacts,sources=sources))
    print(conclusion);print('Report:',OUT/'report.md');print('Artifacts:',len(artifacts))


if __name__=='__main__':main()
