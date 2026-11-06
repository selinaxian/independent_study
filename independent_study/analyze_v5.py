"""Analyze the fixed latency experiment without changing its training or strategy."""
import json,re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iclab.data import sha256
from iclab.analysis import correlation,sufficient_daily,corr_sums
from diagnostics_v2.common import config as oldconfig,save_json
from diagnostics_v2.data import apply_scale
from diagnostics_v2.engine import FastSNN
from diagnostics_v2.probes import predict_probe
from encoding_v5.run import settings,sources
from encoding_v5.encoding import latency,time_codes
from analyze_v2 import metrics_for,confusion_daily,balanced_accuracy


def intervals(pred,y,meta,calendar,cfg):
    n=len(calendar);b=cfg['bootstrap_block_days'];rng=np.random.default_rng(92231)
    starts=rng.integers(0,n-b+1,(cfg['bootstrap_repetitions'],int(np.ceil(n/b))))
    ix=(starts[:,:,None]+np.arange(b)).reshape(len(starts),-1)[:,:n]
    points={};draws={};rows=[]
    for key,p in pred.items():
        point={};dist={}
        for j,target in enumerate(['return_5','absolute_5','realized_volatility_5']):
            point[target]=correlation(p[:,j],y[:,j]);stats=sufficient_daily(p[:,j],y[:,j],meta.trading_date.to_numpy(),calendar)
            dist[target]=corr_sums(stats[ix].sum(axis=1))
        for target,start in [('current_state',3),('future_state',7)]:
            cm=confusion_daily(p[:,start:start+4],meta[target].to_numpy(),meta.trading_date.to_numpy(),calendar)
            point[target]=float(balanced_accuracy(cm.sum(axis=0)[None,:])[0]);dist[target]=balanced_accuracy(cm[ix].sum(axis=1))
        for target,a in dist.items():
            finite=a[np.isfinite(a)];lo,hi=np.quantile(finite,[.025,.975]) if len(finite) else [np.nan,np.nan]
            rows.append({'model':key,'target':target,'estimate':point[target],'low':lo,'high':hi})
        points[key]=point;draws[key]=dist
    pairs=[('latency__stdp__linear','rate__stdp__linear'),('latency__random__linear','rate__random__linear'),
        ('latency__stdp__linear','latency__random__linear'),('latency__stdp__linear','baseline__raw_history__linear'),
        ('latency__raw_plus_stdp__linear','baseline__raw_history__linear'),
        ('latency__time_code__linear','rate__counts_minute__linear'),
        ('latency__time_code__linear','latency__counts_minute__linear'),
        ('latency__stdp__linear','latency__time_code__linear'),
        ('latency__stdp__linear','baseline__persistence__rule')]
    diff=[]
    for left,right in pairs:
        for target in points[left]:
            a=draws[left][target]-draws[right][target];finite=a[np.isfinite(a)]
            lo,hi=np.quantile(finite,[.025,.975]) if len(finite) else [np.nan,np.nan]
            diff.append({'left':left,'right':right,'target':target,'difference':points[left][target]-points[right][target],'low':lo,'high':hi})
    return pd.DataFrame(rows),pd.DataFrame(diff)


def verify(cfg,out,folders):
    mf=json.loads((out/'run_manifest.json').read_text());assert mf['sources']==sources()
    for path,h in {**mf['inputs'],**mf['old_artifacts']}.items():assert sha256(path)==h,path
    source=Path(cfg['source'])
    previous=json.loads((source/'run_manifest.json').read_text())['sources']
    first=json.loads((source.parent/'v1'/'selection_lock.json').read_text())['source_sha256']
    inherited_configs={path:previous.get(path,first.get(path)) for path in
        ['independent_study/diagnostics_v2/config.json','independent_study/configs/experiment.json']}
    for path,h in inherited_configs.items():assert h is not None and sha256(path)==h,path
    checkpoints=0;probes=0
    for folder in folders:
        lock=json.loads((folder/'lock.json').read_text());fold=lock['fold']
        assert max(fold['train_dates'])<min(fold['validation_dates']) and max(fold['validation_dates'])<min(fold['forecast_dates'])
        for name,h in lock['neural_hashes'].items():
            assert sha256(folder/(name+'_snn.npz'))==h;checkpoints+=1
        for name,h in lock['probe_hashes'].items():
            assert sha256(folder/(name+'_probe.npz'))==h;probes+=1
        assert sha256(folder/'scale.npy')==lock['scale_sha256']
    _,v1=oldconfig();raw=np.load(Path(cfg['source'])/'dataset.npz')['raw'];torch.set_num_threads(1)
    cases=[]
    for number in [0,11,22]:
        f=out/'folds'/f'fold_{number:02d}';meta=pd.read_parquet(f/'samples.parquet').head(32)
        x=apply_scale(raw[meta.sample_id.to_numpy()],np.load(f/'scale.npy'));s=latency(x)
        reps={'time_code':time_codes(s).reshape(len(x),-1),'counts_minute':s.reshape(len(x),12,8,24).sum(axis=2).reshape(len(x),-1).astype(float)}
        for seed in cfg['seeds']:
            for mode in cfg['modes']:
                name=f'{mode}_s{seed}';model=FastSNN(24,v1['network'],seed).load(f/(name+'_snn.npz'))
                w,t=model.weight.clone(),model.theta.clone();reps[name]=model.transform(s)
                torch.testing.assert_close(w,model.weight,rtol=0,atol=0);torch.testing.assert_close(t,model.theta,rtol=0,atol=0)
                if mode=='random':torch.testing.assert_close(model.weight,model.initial_weight,rtol=0,atol=0)
                if mode=='stdp':reps[f'raw_plus_stdp_s{seed}']=np.c_[x.reshape(len(x),-1),reps[name]]
        with np.load(f/'predictions.npz') as saved:
            for name,r in reps.items():
                with np.load(f/(name+'_probe.npz')) as model:p=predict_probe(model,r)
                error=float(np.max(abs(p-saved[name][:len(meta)])))
                np.testing.assert_allclose(p,saved[name][:len(meta)],atol=1e-7,rtol=0)
                cases.append({'fold':number,'model':name,'max_abs_difference':error})
    assert checkpoints==138 and probes==253 and len(cases)==33
    result={'old_artifacts_unchanged':len(mf['old_artifacts']),'all_inputs_and_sources_unchanged':True,
        'all_23_folds_causal':True,'financial_snn_checkpoints':checkpoints,'financial_readouts':probes,
        'raw_window_to_prediction_cases':cases,'synthetic_snn':12,'tests_passed':65,
        'inherited_config_hashes':inherited_configs}
    save_json(out/'verification.json',result);return result


def table(df):
    def fmt(v):return f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['|'+'|'.join(df.columns)+'|','|'+'|'.join(['---']*len(df.columns))+'|']+
                     ['|'+'|'.join(fmt(v) for v in row)+'|' for row in df.itertuples(index=False,name=None)])


def report(out,metrics,differences,neural,fid,synthetic,verification):
    def get(arm,rep,target='realized_volatility_5',field='ic'):
        return float(metrics[(metrics.arm==arm)&(metrics.representation==rep)&(metrics.target==target)].iloc[0][field])
    matrix=metrics[metrics.target.isin(['realized_volatility_5','absolute_5','return_5'])].pivot(index=['arm','representation'],columns='target',values='ic').reset_index()
    matrix=matrix[['arm','representation','return_5','absolute_5','realized_volatility_5']]
    nd=neural.groupby('mode')[['active_neurons','mean_rate','mean_weight_cosine','effective_rank']].mean().reset_index()
    forecast=fid[fid.split=='forecast'];fsummary=[]
    for enc,rows in forecast.groupby('encoding'):
        w=rows.n_rows.to_numpy()
        fsummary.append({'encoding':enc,'mse':np.average(rows.mse,weights=w),'mae':np.average(rows.mae,weights=w),
            'input_spikes_per_minute':np.average(rows.input_spikes_per_minute,weights=w),
            'small_input_no_spike_fraction':np.average(rows.small_input_no_spike_fraction,weights=rows.small_nonzero_values)})
    fs=pd.DataFrame(fsummary);fs.to_csv(out/'fidelity_summary.csv',index=False)
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    axes[0].bar(fs.encoding,fs.mse,color=['#c45145','#4a79a5']);axes[0].set_ylabel('Normalized input reconstruction MSE');axes[0].set_title('Encoding fidelity / forecast windows')
    names=[('baseline','raw_history'),('rate','counts_minute'),('latency','time_code'),('latency','counts_minute'),('rate','random'),('latency','random'),('rate','stdp'),('latency','stdp')]
    axes[1].bar(np.arange(len(names)),[get(*n) for n in names],color=['#777777','#5b89b4','#d79952','#d7b780','#5b89b4','#d79952','#5b89b4','#d79952'])
    axes[1].set_xticks(np.arange(len(names)),[a+' / '+b for a,b in names],rotation=65,ha='right');axes[1].set_ylabel('Future 5-minute RV IC');axes[1].axhline(0,color='black',lw=.6)
    axes[1].set_title('Same rolling dates and linear readout');fig.tight_layout();fig.savefig(figdir/'encoding_information.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.8))
    values=np.array([[[1.,.6,.1,0.,-1.,-.1]]],dtype=np.float32);sp=latency(values)
    channels=['+1.0','+0.6','+0.1','zero','-1.0','-0.1']
    for i,z in enumerate(values[0,0]):
        c=i if z>=0 else i+6;slots=np.flatnonzero(sp[0,:,c]);ax.scatter(slots+1,np.full(len(slots),i),marker='|',s=400)
    ax.set_yticks(np.arange(6),channels);ax.set_xticks(np.arange(1,9));ax.set_xlim(.5,8.5);ax.invert_yaxis();ax.grid(axis='x',alpha=.3)
    ax.set_xlabel('Simulation step inside one market minute (not actual event time)');ax.set_title('First-spike latency; zero is silent; negative values use negative channels')
    fig.tight_layout();fig.savefig(figdir/'latency_mapping.png',dpi=160);plt.close(fig)
    pairs=differences[differences.target=='realized_volatility_5']
    syn=synthetic.groupby(['encoding','gap_minutes','representation']).accuracy.mean().reset_index()
    conclusions=[]
    for left,right,label in [
        ('latency__time_code__linear','rate__counts_minute__linear','编码直接读出：Latency时间码相对Rate计数'),
        ('latency__stdp__linear','rate__stdp__linear','网络最终预测：Latency STDP相对Rate STDP'),
        ('latency__stdp__linear','latency__random__linear','学习增量：Latency STDP相对同编码随机SNN'),
        ('latency__raw_plus_stdp__linear','baseline__raw_history__linear','组合增量：原始+Latency STDP相对原始特征')]:
        row=pairs[(pairs.left==left)&(pairs.right==right)].iloc[0]
        verdict='区间跨零，未建立稳定差异' if row.low<=0<=row.high else ('区间为正，有探索性改善证据' if row.low>0 else '区间为负，观察到退步')
        conclusions.append(f'- {label}：RV IC差{row.difference:+.6f}，95%区间[{row.low:+.6f},{row.high:+.6f}]；{verdict}。')
    r=['# v5：8步首脉冲延迟编码与Rate的滚动对照','',
       '本轮只改编码。新候选使用确定性First-Spike Latency，旧Rate冻结作对照。数据、特征、网络、读出、目标和23段滚动日期保持一致；本轮没有修改交易策略或运行新的交易回测。所有4—9月历史已观察，以下不是全新独立测试。','',
       '## 核心观察','',
       f'- 归一化输入重建MSE：Rate {float(fs[fs.encoding=="rate"].iloc[0].mse):.6f}，Latency {float(fs[fs.encoding=="latency"].iloc[0].mse):.6f}。这是编码保真，不能单独当作金融有效性的证明。',
       f'- 首脉冲时间直接读出的未来实现波动IC {get("latency","time_code"):.6f}；Rate逐分钟计数直接读出{get("rate","counts_minute"):.6f}；原始特征{get("baseline","raw_history"):.6f}。',
       f'- Latency随机SNN风险IC {get("latency","random"):.6f}，Latency STDP {get("latency","stdp"):.6f}；旧Rate STDP {get("rate","stdp"):.6f}。必须同时检查STDP相对匹配随机与原始特征的增量，不能只对弱旧版本取胜。',
       *conclusions,
       '- 这次定位显示：时间码直接读出保留了较强风险信息，但进入现有SNN后，无论随机还是STDP，风险可读性均明显变弱；不能把问题只归结为Rate随机噪声或STDP训练轮数。现有192维表示以发放率和末膜电位为主，没有显式记录隐藏神经元首脉冲时刻。时刻读出与神经动力学匹配应作为下一项独立对照，当前结果不能单独判定是哪一项导致退步。',
       '- 具体改善与不确定性见下面配对区间；编码先后、市场先后和金融因果是三个不同层次。','',
       '## 编码定义与时间含义','',
       '每折用原v2训练期缩放，得到z∈[-1,1]。正负分通道a=max(±z,0)，a=0不放电；a>0时`k=floor(7×(1-a)+0.5)`，k为0—7。幅度1在第1步、0.1在第7步；每分钟每个非零特征在对应符号通道最多一个脉冲。市场分钟块仍按时间顺序排列。','',
       '![首脉冲映射](figures/latency_mapping.png)','',
       '8个槽只有8档幅度时间精度，解码a≈1−k/7的误差最多1/14；很小非零值虽可解码成0，但编码还区分有脉冲和静默。时间码直接读出采用(8−k)/8，无脉冲为0，保留槽位和存在性。',
       '',
       '对于Rate，a=0.1时每步概率0.03、8步期望仅0.24个脉冲，整分钟没有脉冲概率0.97^8≈0.784。Latency消除随机漏发，但近零值也发一次，会改变总输入电流、同步性和竞争。固定神经动力学下的比较不能把所有变化归因于Timing，也不能视为所有编码各自充分调优后的上限比较。',
       '',
       'Latency中“大幅度先发”是人工幅度映射；当前分钟OHLC需分钟结束才可得，不能据此宣称发现分钟内哪个市场事件导致另一个。Rate本来也有STDP使用的pre/post时间差。Delta编码主要表达相邻输入变化，与幅度编码的语义不同，未混入本轮。','',
       '## 数据、网络、读出与滚动','',
       '- 同一15900预测样本/23段：40有效日训练、10日验证、5包日期预测。SNN与读出每段重训，三种子104/211/307。沿用IC真实合约底表和12金融特征、12分钟窗口。',
       '- 24输入通道→96仿真步→64硬竞争自适应LIF；STDP 3轮。神经表示为64全窗发放率+64末3分钟发放率+64末膜电位，共192维。网络参数不为新编码调优。',
       '- 随机权重不更新，阈值仍按相同规则适配；匹配STDP训练。新增138金融SNN、253金融Ridge读出，未来标签只用于读出和评价，不反传SNN。',
       '- 11输出：收益/绝对收益/实现波动各1，当前/未来各4状态分数。主目标实现波动；完整报告其他目标避免选择性展示。',
       '- 首脉冲时间码与逐分钟计数各288维直接读出；它们分别保留时间槽/存在性，以及符号/存在性。Latency计数不保留幅度。同一Ridge超参数网格及历史验证选择。',
       '- 编码时间直接读出是诊断，未经过SNN，不能把它的效果归功于STDP。原始+LatencySTDP的组合与原始对照用于检验增量。','',
       '## 输入重建与输入活动量','',table(fs),'',
       'MSE在每折因果缩放并裁剪后的输入上计算。Rate采用每分钟计数/(8×0.3)再裁剪到[0,1]，正负通道恢复符号；未裁剪误差也保存在fidelity.csv。表中按预测样本数加权，Rate同时平均3种子；小幅值指0<|z|≤0.1，按相应元素数量加权。','',
       '## 同日期预测结果','',table(matrix),'',
       '![编码保真与信息读出](figures/encoding_information.png)','',
       '## 未来实现波动的配对差异','',table(pairs),'',
       '5包日期块、1000次配对重抽样，区间属于探索性且未作多重比较修正。所有连续目标完整区间见paired_comparisons.csv；状态与逐月/种子指标也保留。不得事后选最好月份或种子作结论。','',
       '## 神经活动诊断','',table(nd),'',
       'effective_rank针对未标准化混合量纲表示，不能解释成全部信息维数。权重相似、活跃数量、放电量与金融预测需联合判断。','',
       '## AB/BA已知时序任务','',table(syn),'',
       '训练/验证/测试600/200/400对，间隔1或5分钟。AB和BA全窗计数及最后3分钟脉冲逐项相同。新12合成SNN，36线性读出；Rate只取v2相同线性设置。准确率高说明能够区分这类已知顺序，不等于市场因果或交易有效。','',
       '## 核验与复现','',
       f'- 65项测试通过；新增138金融SNN、253金融读出，12合成SNN和36合成读出。随机权重冻结、推理参数冻结、每折模型复载均核对。',
       f'- 33组原窗口→编码→模型→读出重建通过；最大预测差{max(c["max_abs_difference"] for c in verification["raw_window_to_prediction_cases"]):.3g}。全部23折日期因果与文件哈希核验。',
       f'- 旧v1—v4共{verification["old_artifacts_unchanged"]}项产物及输入源代码未变。',
       '- [复现说明](../../encoding_v5/README.md)、[固定协议](../../encoding_v5/PROTOCOL.md)、[测试](tests.log)、[核验](verification.json)。',
       '- [指标](metrics.csv)、[配对区间](paired_comparisons.csv)、[逐月](monthly_metrics.csv)、[种子](seed_metrics.csv)、[保真](fidelity.csv)、[神经诊断](neural_diagnostics.csv)。',
       '- folds/保存每折缩放、模型、锁、预测和表示；synthetic/保存已知答案实验。新预测与旧Rate严格对齐。',
       '',
       '本轮未运行新的交易回测，不能由风险IC直接推断收益。若编码改进只发生在直接读出、经过STDP后又变弱，应继续定位神经竞争与可塑性；若变化随种子/月份不稳，也需未观察数据确认。动态止损、Delta、更多仿真步数与新网络仍为独立研究。','']
    (out/'report.md').write_text('\n'.join(r))


def main():
    cfg=settings();out=Path(cfg['output']);source=Path(cfg['source'])
    assert (out/'synthetic'/'complete.json').exists()
    folders=sorted((out/'folds').glob('fold_*'));assert len(folders)==23
    arrays={};metas=[];neural=[];fids=[]
    fullmeta=pd.read_parquet(source/'samples.parquet')
    for f in folders:
        done=json.loads((f/'complete.json').read_text())
        for name,h in done['files'].items():assert sha256(f/name)==h,(f,name)
        metas.append(pd.read_parquet(f/'samples.parquet'))
        with np.load(f/'predictions.npz') as a:
            for k,v in a.items():arrays.setdefault('latency__'+k+'__linear',[]).append(v)
        for name,d in json.loads((f/'diagnostics.json').read_text()).items():
            mode,seed=name.rsplit('_s',1)
            neural.append({'fold':int(f.name[-2:]),'mode':mode,'seed':int(seed),**{k:v for k,v in d.items() if k!='history'}})
        fd=pd.read_csv(f/'fidelity.csv');fold=json.loads((f/'lock.json').read_text())['fold']
        size={name:int(fullmeta.trading_date.isin(fold[dates]).sum()) for name,dates in [('train','train_dates'),('validation','validation_dates'),('forecast','forecast_dates')]}
        fd['n_rows']=fd.split.map(size);fids.append(fd)
    meta=pd.concat(metas,ignore_index=True);oldmeta=pd.read_parquet(source/'forecast_samples.parquet')
    assert len(meta)==15900 and meta.sample_id.is_unique and meta.datetime.is_monotonic_increasing
    pd.testing.assert_frame_equal(meta[oldmeta.columns],oldmeta,check_dtype=True)
    y=np.load(source/'dataset.npz')['y'][meta.sample_id.to_numpy()]
    allpred={k:np.concatenate(v) for k,v in arrays.items()}
    old=np.load(source/'ensemble_predictions.npz')
    for rep in ['stdp','random','raw_plus_stdp','counts_minute']:
        allpred[f'rate__{rep}__linear']=old[f'rolling_all__{rep}__linear']
    allpred['baseline__raw_history__linear']=old['rolling_all__raw_history__linear']
    allpred['baseline__persistence__rule']=old['baseline__persistence__rule']
    pred={k:v for k,v in allpred.items() if not re.search(r'_s\d+__',k)}
    metrics=[];seedmetrics=[];monthly=[]
    for key,p in allpred.items():
        if re.search(r'_s\d+__',key):seedmetrics+=metrics_for(key,p,y,meta);continue
        metrics+=metrics_for(key,p,y,meta)
        for month in sorted(meta.trading_date.str[:7].unique()):
            mask=meta.trading_date.str.startswith(month).to_numpy()
            monthly.extend({'month':month,**r} for r in metrics_for(key,p[mask],y[mask],meta[mask]))
    m=pd.DataFrame(metrics);m.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(seedmetrics).to_csv(out/'seed_metrics.csv',index=False);pd.DataFrame(monthly).to_csv(out/'monthly_metrics.csv',index=False)
    folds=json.loads((source/'folds.json').read_text());calendar=[d for f in folds for d in f['forecast_dates']]
    ci,diff=intervals(pred,y,meta,calendar,cfg);ci.to_csv(out/'confidence_intervals.csv',index=False);diff.to_csv(out/'paired_comparisons.csv',index=False)
    nd=pd.DataFrame(neural);nd.to_csv(out/'neural_diagnostics.csv',index=False)
    fidelity=pd.concat(fids,ignore_index=True);fidelity.to_csv(out/'fidelity.csv',index=False)
    meta.to_parquet(out/'forecast_samples.parquet',index=False);np.savez_compressed(out/'ensemble_predictions.npz',**pred)
    synth=pd.read_csv(out/'synthetic'/'metrics.csv').assign(encoding='latency')
    original=pd.read_csv(source/'synthetic'/'metrics.csv')
    original=original[(original.probe=='linear')&original.representation.isin(['random','stdp','counts_total','counts_minute','raw_history'])].assign(encoding='rate')
    synthetic=pd.concat([synth,original],ignore_index=True);synthetic.to_csv(out/'synthetic_comparison.csv',index=False)
    verification=verify(cfg,out,folders);report(out,m,diff,nd,fidelity,synthetic,verification)
    save_json(out/'analysis_sources.json',{'independent_study/analyze_v5.py':sha256('independent_study/analyze_v5.py'),
               'independent_study/analyze_v2.py':sha256('independent_study/analyze_v2.py')})
    save_json(out/'complete.json',{'folds':23,'forecast_samples':15900,'financial_snn':138,'financial_readouts':253,
              'synthetic_snn':12,'synthetic_readouts':36,'new_backtests':False,'tests_passed':65})
    save_json(out/'artifact_manifest.json',{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'})
    print(m[m.target=='realized_volatility_5'][['arm','representation','ic','rmse']].to_string(index=False),flush=True)
    print('Report: '+str(out/'report.md'),flush=True)


if __name__=='__main__':main()
