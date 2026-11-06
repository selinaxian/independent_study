"""Write the auditable Chinese application report and standalone figures."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iclab.data import sha256
from diagnostics_v2.common import save_json
from .prepare import settings,ROOT


NAMES={'baseline':'无风险层','stdp__rv':'STDP实现波动','stdp__absolute':'STDP绝对收益',
       'random__rv':'随机SNN实现波动','random__absolute':'随机SNN绝对收益',
       'raw_history__rv':'原始特征实现波动','raw_history__absolute':'原始特征绝对收益',
       'raw_plus_stdp__rv':'原始+STDP实现波动','raw_plus_stdp__absolute':'原始+STDP绝对收益',
       'persistence__rv':'当前波动持续','persistence__absolute':'当前绝对收益持续'}


def table(rows,columns):
    result=['|'+'|'.join(label for key,label in columns)+'|','|'+'|'.join(['---']*len(columns))+'|']
    for r in rows.to_dict('records'):
        values=[]
        for key,_ in columns:
            v=r[key]
            if key=='risk':v=NAMES.get(v,v)
            elif isinstance(v,(float,np.floating)):v='—' if not np.isfinite(v) else f'{v:,.3f}'
            values.append(str(v))
        result.append('|'+'|'.join(values)+'|')
    return '\n'.join(result)


def report():
    cfg=settings();out=Path(cfg['output']);s=pd.read_csv(out/'summary.csv');daily=pd.read_csv(out/'daily.csv')
    verification=json.loads((out/'verification.json').read_text());prepared=json.loads((out/'prepared.json').read_text())
    figures=out/'figures';figures.mkdir(exist_ok=True)
    colors={'baseline':'#777777','stdp__rv':'#c74440','stdp__absolute':'#e9a044','raw_history__rv':'#3b76af','persistence__rv':'#36886f'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,application in zip(axes,['filter','sizing']):
        for name,color in colors.items():
            q=daily[(daily.direction=='raw_history')&(daily.application==application)&(daily.risk==name)&(daily.ticks==1)]
            base=1e6 if application=='filter' else 2e6
            ax.plot(pd.to_datetime(q.trading_date),(q.equity/base-1)*100,label=name,color=color,lw=1.4)
        ax.axhline(0,color='black',lw=.5);ax.set_title(application+' / fixed raw direction');ax.set_ylabel('Net return (%)')
        ax.grid(alpha=.2);ax.tick_params(axis='x',rotation=30)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(figures/'risk_equity.png',dpi=160);plt.close(fig)
    q=s[(s.direction=='raw_history')&(s.application=='filter')&(s.ticks==1)&s.risk.isin(colors)&(s.risk!='baseline')]
    fig,ax=plt.subplots(figsize=(8,4.5))
    x=np.arange(len(q));point=q.selection_excess.to_numpy()/1e4
    ax.vlines(x,q.selection_excess_low.to_numpy()/1e4,q.selection_excess_high.to_numpy()/1e4)
    ax.scatter(x,point)
    ax.axhline(0,color='black',lw=1);ax.set_xticks(x,q.risk,rotation=15)
    ax.set_ylabel('Net P&L above matched exposure (CNY 10,000)');ax.set_title('Exploratory paired 5-day block intervals')
    ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(figures/'selection_excess.png',dpi=160);plt.close(fig)
    findings=[]
    main=s[(s.direction=='raw_history')&(s.ticks==1)]
    for application,label in [('filter','开仓过滤（100万本金）'),('sizing','整数仓位（200万本金）')]:
        a=main[(main.application==application)&(main.risk=='stdp__rv')].iloc[0]
        b=main[(main.application==application)&(main.risk=='baseline')].iloc[0]
        p=main[(main.application==application)&(main.risk=='persistence__rv')].iloc[0]
        findings.append(f'- {label}：STDP实现波动控制后的净收益{a.net_pnl/1e4:.2f}万元，无风险层为{b.net_pnl/1e4:.2f}万元；最大回撤绝对值从{abs(b.max_drawdown_pct):.2f}%降至{abs(a.max_drawdown_pct):.2f}%，总手数保留{a.lot_fraction*100:.1f}%。同暴露随机期望以上的差额为{a.selection_excess/1e4:.2f}万元，95%日期块区间[{a.selection_excess_low/1e4:.2f},{a.selection_excess_high/1e4:.2f}]万元，跨零，尚未建立额外择机优势。简单当前波动持续基线净收益{p.net_pnl/1e4:.2f}万元。')
    findings.extend([
        '- STDP绝对收益过滤主底座净亏30.94万元，弱于STDP实现波动过滤的点估计；辅助selected底座甚至由净亏24.80万元变为净亏29.42万元。终点幅度不是通用止损风险代理。',
        '- 当前波动持续的过滤主底座超过同暴露随机期望9.83万元，区间[1.10,19.50]万元；辅助底座也有正差额。本轮可以看到简单波动过滤的有限积极证据，但自身仍净亏，且这些是已观察历史、未校正多重比较的探索结果。',
        f'- 全部{len(s)}个预设组合（含基线、两方向、两成本）中，净盈利组合为{int((s.net_pnl>0).sum())}。主底座、单边1tick下，STDP实现波动与随机/原始/持续风险来源的配对收益差区间均跨零，不能因某个点估计更好而宣称STDP增量成立。',
        '- 在672个主底座机会中，STDP波动预测与未来RV的IC约0.074、与方向MAE约0.069；当前波动持续基线分别约0.703和0.489。STDP过滤掉的207笔实现RV均值22.65bp，保留的465笔19.14bp，但被过滤机会平均单手净亏344元、保留机会净亏579元：过滤较高波动不等于剔除了最差净收益机会。',
        '- 输出用途上优先RV→风险预算/仓位和过滤仍合理；当前实验不足以证明STDP比简单波动信息更合适。后续改善表示信息保留与风险校准，应独立于本版本应用参数，不能事后重选门槛来把本轮改成赢家。'
    ])
    cols=[('risk','风险来源'),('trades','笔数'),('lots','总手数'),('lot_fraction','相对手数'),
          ('net_pnl','净收益元'),('max_drawdown_pct','最大回撤%'),('daily_loss_es95','日损失ES95元'),
          ('selection_excess','超过同暴露随机期望元'),('selection_excess_low','差额区间下界'),('selection_excess_high','差额区间上界')]
    r=['# v4：风险预测的开仓过滤与整数仓位实验','',
       '本轮将风险用途单独检验，复用v2全滚动SNN和Ridge读出。没有重新训练、改编码或加入止损。2026年4—9月已在前轮研究观察，以下是历史滚动探索，不是新独立测试。','',
       '## 已观察成果','',*findings,'',
       '## 哪种输出适合什么用途','',
       '|输出|适用用途|本轮处理|','|---|---|---|',
       '|未来5分钟实现波动|路径起伏、风险预算与仓位调整；可检验开仓过滤|主目标|',
       '|未来5分钟绝对收益|终点位移幅度；可能漏掉先跌后涨的风险|完整对照|',
       '|当前/未来四状态|只有状态预测可靠、与策略收益对应时才可分状态交易|本轮不接入；v2未显示增量优势|',
       '|未来5分钟收益|决定多空方向|方向底座保持既有预测|',
       '|方向条件最大不利波动/触及阈值概率|更直接对应止损|当前没有该读出；未实施|','',
       '高波动不是高亏损概率。高波动也可能提供收益机会；降低波动暴露不自动增加净收益。绝对收益是终点幅度，RV是分钟采样路径的平方对数收益和开根号，仍无法完整描述分钟内部尾部走势。','',
       '## 数据、模型与因果流程','',
       '- 沿用明确IC合约分钟数据、12金融特征、12分钟窗口、24正负通道、8仿真步/分钟、64竞争LIF、192维表示、监督Ridge读出；3种子平均。STDP只做无监督表示。',
       '- SNN和读出均来自v2既定40日训练/10日验证/5包日期预测，共23段、15900预测；每段重新初始化训练，非在线续训。本轮只加载这些检查点推理。',
       '- 过滤和仓位所需标尺只用各段过去10日验证风险预测的分布，未来风险/收益不参与门槛或手数。验证期也曾用于Ridge超参数，不把其成绩当独立证明。',
       '- 主方向底座为原始特征收益读出，固定既有方向和入场门槛；原底座属于诊断，允许最佳验证收益为负时交易。辅助selected底座按过去验证择原始/STDP/混合或空仓。',
       '- 固定原机会时钟：跳过一笔后仍等待原虚拟5分钟持有结束再考虑下一笔，控制因为开仓时间变化引入的差异。没有持仓中的动态调仓。','',
       '## 固定风险规则与成交核算','',
       '过滤：风险预测≤过去验证预测80%分位才开1手，否则0手。基准固定1手、100万元本金。',
       '',
       '仓位：`min(2, floor(2 × max(过去验证预测中位数,1bp) / max(当前风险预测,1bp)))`，只允许0/1/2手。基准固定2手、200万元本金，与单手方案初始杠杆一致；不同本金方案应比较收益率/回撤，而非直接比较亏损金额。它是相对风险缩仓，并非校准VaR。',
       '',
       '门槛、上限和下限在运行前固定，没有搜索本轮收益最高的参数。所有组合均完整保留。方向信号后留一整分钟，开盘价模拟入场、持有5分钟、不跨午休/隔夜；乘数200、tick0.2、原手续费、单边1tick和2tick敏感性；订单固定费用每单一次，按实际手数计比例费用。资金连续，保证金约束实际检查。','',
       '## 主底座：固定原始特征方向','',
       '### 开仓过滤（100万元，单边1tick）','',
       table(s[(s.direction=='raw_history')&(s.application=='filter')&(s.ticks==1)],cols),'',
       '### 整数仓位（200万元，单边1tick）','',
       table(s[(s.direction=='raw_history')&(s.application=='sizing')&(s.ticks==1)],cols),'',
       '![累计净收益率](figures/risk_equity.png)','',
       '## 如何区分预测有效与单纯少交易','',
       '同暴露随机控制在每个滚动段固定机会中打乱0/1/2手分配，保留该段实际交易笔数和总手数。其期望净收益是同样交易量随机选择机会的参照。真实净收益减这个期望，才更接近风险排序是否选对机会的检验。该控制使用事后暴露做归因，不能部署，也不参与实际规则选择。',
       '',
       '表中差额区间使用5包日期块、1000次重抽样；同时保存1000次手数打乱参照。区间属于探索性分析，未作多重比较修正，不应事后从多个候选中挑一个称正式赢家。随机参照不复算路径保证金；实际保证金拒绝情况见核验。少亏、回撤降低、净盈利、超过简单风险基线是不同结论。空仓始终是0收益/0交易风险对照。','',
       '![同暴露差额](figures/selection_excess.png)','',
       '## 辅助selected方向底座（单边1tick）','',
       table(s[(s.direction=='selected')&(s.application=='filter')&(s.ticks==1)],cols),'',
       '辅助底座部分时段本身使用STDP方向，因此主归因以原始方向底座为准。原始+STDP风险读出也不等于STDP单独有效；需要与原始风险读出相比较。','',
       '## 滑点敏感性','',
       table(s[(s.direction=='raw_history')&(s.ticks==2)&s.risk.isin(['baseline','stdp__rv','stdp__absolute','raw_history__rv','persistence__rv'])],
             [('application','应用'),('risk','风险来源'),('net_pnl','2tick净收益元'),('return_pct','收益率%'),('max_drawdown_pct','最大回撤%')]),'',
       '## 核验与可复现产物','',
       f'- 新增6项测试及既有53项回归，记录见[tests.log](tests.log)。23段重建预测最大绝对误差为{max(f["reproduction_max_abs_diff"] for f in prepared["folds"]):.3g}。',
       f'- 共{verification["cases"]}种应用/基线/方向/成本组合；全部整数手数；单手不加风险层逐分钟净值与旧回测一致（最大误差见verification.json）。最大保证金拒绝数{verification["max_margin_rejections"]}。',
       f'- 已核对旧版本{verification["old_artifacts_unchanged"]}项产物哈希未变。未改原数据、模型或旧报告；本轮代码/协议与原模型来源单独记录。',
       '- [summary.csv](summary.csv)：全部净收益、回撤、ES、同暴露比较。',
       '- [monthly.csv](monthly.csv)：逐月损益与选择差额；[paired_comparisons.csv](paired_comparisons.csv)：STDP与其他风险来源的配对收益差。',
       '- [opportunity_prediction_metrics.csv](opportunity_prediction_metrics.csv)：真实机会上的RV、绝对收益和方向MAE关联；[kept_skipped_diagnostics.csv](kept_skipped_diagnostics.csv)：保留/过滤机会的实现风险。',
       '- calibration/保存每折历史预测、未来预测、门槛和标尺；backtests/保存每笔机会（含0手）、逐日与逐分钟资金。',
       '- [verification.json](verification.json)、[prepared.json](prepared.json)、[source_lock.json](source_lock.json)记录重建与完整性核验。','',
       '运行：`PYTHONPATH=independent_study OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=1 .venv/bin/python -m risk_v4.prepare`，再运行`risk_v4.run`、`risk_v4.report`。不得把旧结果目录覆盖作为新实验。','',
       '## 限制与后续边界','',
       '行情分钟标签、供应商价格调整/数量单位等仍未获确认，成交按分钟开盘价加滑点模拟；这里不能推断真实挂单成交概率。没有tick盘口，也没有止损触发/成交实验。',
       '',
       '止损后续应独立学习持仓方向条件的最大不利波动、损失分位数或触及阈值概率，并核实分钟内触价与成交规则。不能把未来RV预测乘一个常数便称已验证的止损模型，也不能简单把高波动对应更紧止损。新参数、模型或交易用途的最终确认需要未观察数据。','']
    (out/'report.md').write_text('\n'.join(r))
    save_json(out/'report_sources.json',{str(ROOT/'report.py'):sha256(ROOT/'report.py')})
    manifest={str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'}
    save_json(out/'artifact_manifest.json',manifest)
    print(f'report complete: {out}/report.md; {len(manifest)} artifacts',flush=True)


if __name__=='__main__':report()
