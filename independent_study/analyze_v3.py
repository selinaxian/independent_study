"""Analyze structure-only experiments after all rolling predictions are locked."""
from pathlib import Path
import json,re
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iclab.data import sha256
from iclab.neural import encode
from iclab.analysis import sufficient_daily,corr_sums,correlation
from diagnostics_v2.common import save_json
from diagnostics_v2.data import apply_scale
from diagnostics_v2.probes import predict_probe
from architecture_v3.run import settings,sources
from architecture_v3.neural import SoftLayer
from analyze_v2 import metrics_for,confusion_daily,balanced_accuracy,state_details


def table(df):
    def fmt(v):return f'{v:.4f}' if isinstance(v,(float,np.floating)) and np.isfinite(v) else str(v)
    return '\n'.join(['|'+'|'.join(df.columns)+'|','|'+'|'.join(['---']*len(df.columns))+'|']+['|'+'|'.join(fmt(v) for v in row)+'|' for row in df.itertuples(index=False,name=None)])


def intervals_and_differences(pred,y,meta,calendar,cfg):
    n=len(calendar);block=cfg['bootstrap_block_days'];rng=np.random.default_rng(92231)
    starts=rng.integers(0,n-block+1,(cfg['bootstrap_repetitions'],int(np.ceil(n/block))))
    ix=(starts[:,:,None]+np.arange(block)).reshape(len(starts),-1)[:,:n]
    points={};draws={};rows=[]
    for key,p in pred.items():
        points[key]={};draws[key]={}
        for j,target in enumerate(['return_5','absolute_5','realized_volatility_5']):
            points[key][target]=correlation(p[:,j],y[:,j])
            sums=sufficient_daily(p[:,j],y[:,j],meta.trading_date.to_numpy(),calendar)
            draws[key][target]=corr_sums(sums[ix].sum(1))
        for target,start in [('current_state',3),('future_state',7)]:
            cm=confusion_daily(p[:,start:start+4],meta[target].to_numpy(),meta.trading_date.to_numpy(),calendar)
            points[key][target]=float(balanced_accuracy(cm.sum(0)[None,:])[0])
            draws[key][target]=balanced_accuracy(cm[ix].sum(1))
        for target,a in draws[key].items():
            finite=a[np.isfinite(a)]
            rows.append({'model':key,'target':target,'estimate':points[key][target],'ci_low':np.quantile(finite,.025),'ci_high':np.quantile(finite,.975)})
    pairs=[]
    for mode in ['random','stdp']:
        pairs += [(f'soft_adapt_{mode}',f'hard_adapt_{mode}'),(f'soft_adapt_{mode}',f'soft_fixed_{mode}'),(f'deep_last_{mode}',f'soft_adapt_{mode}'),(f'deep_concat_{mode}',f'soft_adapt_{mode}')]
    for architecture in ['hard_adapt','soft_fixed','soft_adapt','deep_last','deep_concat']:
        pairs += [(architecture+'_stdp',architecture+'_random'),(architecture+'_stdp','raw_history'),(architecture+'_stdp','persistence')]
    diff=[]
    for left,right in pairs:
        l=left+'__linear';r=right+('__rule' if right=='persistence' else '__linear')
        for target in draws[l]:
            d=draws[l][target]-draws[r][target];d=d[np.isfinite(d)]
            diff.append({'left':left,'right':right,'target':target,'difference':points[l][target]-points[r][target],'ci_low':np.quantile(d,.025),'ci_high':np.quantile(d,.975)})
    return pd.DataFrame(rows),pd.DataFrame(diff)


def verification(cfg,v1,out,folders):
    mf=json.loads((out/'run_manifest.json').read_text())
    assert mf['sources']==sources()
    assert all(sha256(p)==h for p,h in mf['inputs'].items())
    v2=Path(cfg['v2_dir']);old=json.loads((v2/'artifact_manifest.json').read_text())
    assert all(sha256(v2/p)==h for p,h in old.items())
    count=0
    for f in folders:
        lock=json.loads((f/'lock.json').read_text());fold=lock['fold']
        assert max(fold['train_dates'])<min(fold['validation_dates']) and max(fold['validation_dates'])<min(fold['forecast_dates'])
        for name,h in {**lock['model_files'],**lock['probe_files']}.items():assert sha256(name)==h,name
        assert lock['all_encoder_freeze_reload'] and lock['first_layer_frozen']
        count+=len(lock['model_files'])
    raw=np.load(v2/'dataset.npz')['raw'];reproduced=0;torch.set_num_threads(1)
    for number in [0,11,22]:
        f=out/'folds'/f'fold_{number:02d}';m=pd.read_parquet(f/'samples.parquet').head(24);saved=np.load(f/'predictions.npz')
        lock=json.loads((f/'lock.json').read_text());x=apply_scale(raw[m.sample_id.to_numpy()],np.load(lock['scale_path']));spikes=encode(x,m.sample_id.to_numpy(),'rate',104)
        for mode in ['random','stdp']:
            reps={}
            for architecture,adaptive in [('soft_fixed',False),('soft_adapt',True)]:
                name=architecture+'_'+mode+'_s104';layer=SoftLayer(24,v1['network'],104,adaptive).load(f/(name+'_snn.npz'))
                r,seq=layer.transform_with_spikes(spikes);reps[name]=r
                if adaptive:
                    second=SoftLayer(64,{**v1['network'],'hidden':32},10104).load(f/('deep_layer2_'+mode+'_s104_snn.npz'))
                    r2=second.transform(seq);reps['deep_last_'+mode+'_s104']=r2;reps['deep_concat_'+mode+'_s104']=np.c_[r,r2]
            for name,r in reps.items():
                prediction=predict_probe(np.load(f/(name+'__linear_probe.npz')),r)
                np.testing.assert_allclose(prediction,saved[name+'__linear'][:len(m)],rtol=1e-6,atol=1e-6);reproduced+=1
    assert count==414 and reproduced==24
    result={'all_sources_inputs_and_v2_artifacts_unchanged':True,'all_23_folds_causal':True,'all_model_hashes_freeze_and_reload':True,'new_layer_checkpoints':count,'raw_to_forecast_cases_reproduced':reproduced}
    save_json(out/'verification.json',result)
    return result


def report(cfg,out,metrics,diff,ci,neural,verify):
    def get(rep,target,field='ic'):
        return metrics[(metrics.representation==rep)&metrics.target.eq(target)].iloc[0][field]
    variants=['hard_adapt_random','hard_adapt_stdp','soft_fixed_random','soft_fixed_stdp','soft_adapt_random','soft_adapt_stdp','deep_last_random','deep_last_stdp','deep_concat_random','deep_concat_stdp','raw_history','counts_minute','risk_control','persistence']
    rows=[]
    for rep in variants:
        rows.append({'模型':rep,'收益IC':get(rep,'return_5'),'绝对收益IC':get(rep,'absolute_5'),'实现波动IC':get(rep,'realized_volatility_5'),'当前状态BA':get(rep,'current_state','balanced_accuracy'),'未来状态BA':get(rep,'future_state','balanced_accuracy')})
    ns=neural.groupby(['architecture','mode']).agg(平均权重余弦=('mean_weight_cosine','mean'),平均有效秩=('effective_rank','mean'),平均发放率=('mean_rate','mean'),最少活跃神经元=('active_neurons','min')).reset_index()
    selected=diff[diff.target.isin(['absolute_5','realized_volatility_5','future_state']) & diff.left.str.endswith('stdp')]
    directions=[]
    for left,right in [('soft_adapt_stdp','hard_adapt_stdp'),('soft_adapt_stdp','soft_fixed_stdp'),('deep_last_stdp','soft_adapt_stdp'),('deep_concat_stdp','soft_adapt_stdp')]:
        r=diff[(diff.left==left)&(diff.right==right)&diff.target.eq('realized_volatility_5')].iloc[0]
        evidence='区间高于0' if r.ci_low>0 else ('区间低于0' if r.ci_high<0 else '区间跨0')
        directions.append(f'- {left}相对{right}：未来实现波动IC差{r.difference:.4f}，95%区间[{r.ci_low:.4f}, {r.ci_high:.4f}]，{evidence}。')
    best_raw=get('raw_history','realized_volatility_5')
    text=f'''# v3：软抑制、自适应阈值与双层SNN结构消融

2026-10-28。该版本只回答网络结构问题；数据、特征、编码、11目标、线性读出和23段滚动与v2保持一致。**4—9月已观察，本轮仍是探索性历史检验。**版本与实验分工见[版本索引](../../VERSIONS.md)，上轮完整诊断见[v2报告](../v2/report.md)。

## 主要发现

{chr(10).join(directions)}

原始特征的实现波动IC为{best_raw:.4f}，直接用当前5分钟实现波动预测未来的持续基线IC为{get('persistence','realized_volatility_5'):.4f}。即使某个结构超过原来的STDP，也须检查它是否超过匹配随机网络、原始特征和简单持续性；改善旧网络不等于新增金融信息。没有用本轮结果事后挑一个交易策略。

## 1. 固定项与结构编号

同一15900条预测、106有效日、115包日期；40有效日训练、随后10日验证、后5包日期预测，共23段。3种子104/211/307。12特征→正负24通道→每分钟8仿真步、12分钟96步；rate编码、训练缩放、3轮、批128、学习率和STDP行归一化均固定。11目标是收益、绝对收益、实现波动各1列，当前/未来四状态各4列。未来标签只用于监督Ridge读出和事后评价。

|编号|模型|作用|
|---|---|---|
|N0|hard_adapt，64硬竞争+自适应阈值|复用v2已锁预测|
|N1|soft_adapt，64软抑制+自适应阈值|与N0比较竞争方式|
|N2|soft_fixed，64软抑制+固定阈值|与N1比较阈值作用|
|N3|64→32软抑制+自适应阈值|第一层复用N1并冻结，训练第二层|

N3的deep_last读取第二层96维，deep_concat拼接两层288维；它们共享同一神经模型。单层输出192维，均为全窗发放、末3分钟发放和末态膜电位。每种结构都有相同种子的随机突触对照；随机模型按相同规则适配阈值。新增第二层也增加参数和表示维度，因此结果不完全等同于纯深度效应。

## 2. 机制定义

旧网络已有0.15抑制项和训练期自适应阈值，但每步强制只保留一个winner。本版软抑制允许所有越阈值单元发放，随后对每个神经元减去0.15×其他发放单元数，再截断为非负；该同步侧抑制影响下一步膜电位，不强制最多一次发放。

阈值适配沿用theta←0.999theta+0.03×batch_mean(spike)，固定阈值消融theta始终0。预测期突触和theta均冻结，窗口间膜电位/迹清零；不是新的逐样本快速阈值或在线跨日适应。软抑制改变发放/塑性事件数量，没有匹配发放预算。

第二层确实以第一层64路脉冲作为输入，种子+10000、32神经元、突触行和4.8，第一层在第二层训练中不更新。是逐层无监督STDP，不使用反向传播或未来标签训练神经层。精确参数和公式见[协议](../../architecture_v3/PROTOCOL.md)。

参考本地BindsNET的[DiehlAndCookNodes](../../../bindsnet/bindsnet/network/nodes.py)中训练期阈值与one_spike开关，以及[模型](../../../bindsnet/bindsnet/models/models.py)中循环抑制的组织方式。本项目沿用自己的批平均阈值、同步代数抑制和既有STDP参数；没有导入该库或声称数值复现其模型。

## 3. 相同样本、相同线性读出结果

{table(pd.DataFrame(rows))}

![结构预测比较](figures/prediction.png)

BA是四类平衡准确率，机会水平0.25；当前状态由已有价格定义，机械规则识别为100%，当前成绩不能当成提前预测。绝对收益与实现波动都是风险幅度，不能直接决定买卖方向。

## 4. 配对不确定性

{table(selected)}

区间使用相同115包日期、5日块、1000次配对重抽样；没有重新训练bootstrap模型，也没有多重检验校正。所有模型和三个种子结果保留于metrics.csv、seed_metrics.csv、monthly_metrics.csv、paired_comparisons.csv，不根据本轮成绩重选结构、调参或改变滚动窗口。

## 5. 神经元是否学到了更多不同模式

{table(ns)}

![神经元诊断](figures/neural.png)

放电更多、活跃单元更多或权重变化更大，都不自动代表金融信息更多。权重余弦越接近1，输入权重方向越相似，可提示模式趋同；但这是训练诊断，不是因果证明。有效秩只对原始表示中心化，会受膜电位与发放率量纲影响，不等于可预测信息量。第二层的有效秩和第一层维度不同，也不能只比较绝对数值。

## 6. 版本边界及下一步

本轮已完成用户提出的结构扩展与逐项消融，**没有同时更换编码、特征、风险标签或止损规则**。滚动框架作为固定评估方式沿用。风险用途应另外检验：固定同一基础交易信号，比较无过滤、简单历史波动过滤、原始特征风险读出、STDP风险读出，在过去验证上选门槛，再检验净收益/回撤/尾部损失。动态止损还须处理分钟内价格路径和滑点；挂单成交概率则需适合的盘口/订单数据。

不能把v1绝对收益IC约0.16视为已经稳定确认的风险优势；v2全滚动该值约0.025。研究目标仍是STDP相对可用基线的增量价值，而不只是把最终输出改名为风险分数。新的确认需要更晚日期或其他独立数据。

## 7. 完成与复现

新增4项机制检查和原49项共53项通过。新训练414个层检查点（23段×3种子×2模式×3层配置），每段24个线性读出，共552个；deep_last/deep_concat共用第二层。三个普通进程按fold模3分工，输出互斥，无共享模型状态。

核验结果：{json.dumps(verify,ensure_ascii=False)}。

folds/fold_XX包含层权重、训练锁、读出/验证后锁、预测及表示；run_manifest记录源码与继承输入；verification、analysis_sources和artifact_manifest记录核验。读取和运行见[复现说明](../../architecture_v3/README.md)。v1、v2数据和产物均保留。
'''
    figs=out/'figures';figs.mkdir(exist_ok=True)
    names=['hard_adapt','soft_fixed','soft_adapt','deep_last','deep_concat']
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,target in zip(axes,['return_5','realized_volatility_5','future_state']):
        field='balanced_accuracy' if 'state' in target else 'ic'
        for offset,mode in [(-.18,'random'),(.18,'stdp')]:ax.bar(np.arange(5)+offset,[get(n+'_'+mode,target,field) for n in names],width=.35,label=mode)
        ax.axhline(get('raw_history',target,field),color='black',linestyle='--',label='raw history')
        ax.axhline(get('persistence',target,field),color='gray',linestyle=':',label='persistence')
        ax.set_xticks(np.arange(5),names,rotation=30,ha='right');ax.set(title=target,ylabel=field);ax.legend(fontsize=7)
    fig.savefig(figs/'prediction.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,field in zip(axes,['mean_weight_cosine','mean_rate']):
        grouped=neural.groupby(['architecture','mode'])[field].mean().unstack();grouped.plot.bar(ax=ax,rot=20)
        ax.set(ylabel=field,title='Training diagnostics')
    fig.savefig(figs/'neural.png',dpi=150);plt.close(fig)
    (out/'report.md').write_text(text)


def main():
    cfg,c2,v1=settings();out=Path(cfg['output_dir']);v2=Path(cfg['v2_dir'])
    for i in range(cfg['workers']):assert (out/f'worker_{i}_complete.json').exists()
    folders=sorted((out/'folds').glob('fold_*'));assert len(folders)==23
    metas=[];arrays={};neural=[]
    for f in folders:
        done=json.loads((f/'complete.json').read_text());assert sha256(f/'predictions.npz')==done['predictions_sha256'];assert sha256(f/'lock.json')==done['lock_sha256']
        metas.append(pd.read_parquet(f/'samples.parquet'))
        with np.load(f/'predictions.npz') as a:
            for key in a.files:arrays.setdefault(key,[]).append(a[key])
        for name,d in json.loads((f/'diagnostics.json').read_text()).items():
            stem,seed=name.rsplit('_s',1);architecture,mode=stem.rsplit('_',1)
            neural.append({'fold':int(f.name[-2:]),'architecture':architecture,'mode':mode,'seed':int(seed),**{k:v for k,v in d.items() if k!='history'}})
    meta=pd.concat(metas,ignore_index=True);assert len(meta)==15900 and meta.sample_id.is_unique and meta.datetime.is_monotonic_increasing
    y=np.load(v2/'dataset.npz')['y'][meta.sample_id.to_numpy()]
    allpred={k:np.concatenate(v) for k,v in arrays.items()};pred={k:v for k,v in allpred.items() if not re.search(r'_s\d+__',k)}
    metrics=[];seedrows=[];monthly=[]
    for key,p in allpred.items():
        canonical='rolling_all__'+key
        if re.search(r'_s\d+__',key):seedrows+=metrics_for(canonical,p,y,meta);continue
        metrics+=metrics_for(canonical,p,y,meta)
        for month in sorted(meta.trading_date.str[:7].unique()):
            mask=meta.trading_date.str.startswith(month).to_numpy();monthly += [{'month':month,**r} for r in metrics_for(canonical,p[mask],y[mask],meta[mask])]
    m=pd.DataFrame(metrics);m.to_csv(out/'metrics.csv',index=False);pd.DataFrame(seedrows).to_csv(out/'seed_metrics.csv',index=False);pd.DataFrame(monthly).to_csv(out/'monthly_metrics.csv',index=False)
    folds=json.loads((v2/'folds.json').read_text());calendar=[d for f in folds for d in f['forecast_dates']]
    ci,diff=intervals_and_differences(pred,y,meta,calendar,cfg);ci.to_csv(out/'confidence_intervals.csv',index=False);diff.to_csv(out/'paired_comparisons.csv',index=False)
    state_details({'rolling_all__'+k:v for k,v in pred.items()},meta,out)
    nd=pd.DataFrame(neural);nd.to_csv(out/'neural_diagnostics.csv',index=False)
    meta.to_parquet(out/'forecast_samples.parquet',index=False);np.savez_compressed(out/'ensemble_predictions.npz',**pred)
    checks=verification(cfg,v1,out,folders)
    report(cfg,out,m,diff,ci,nd,checks)
    save_json(out/'analysis_sources.json',{'independent_study/analyze_v3.py':sha256('independent_study/analyze_v3.py'),'independent_study/analyze_v2.py':sha256('independent_study/analyze_v2.py')})
    save_json(out/'complete.json',{'folds':23,'forecast_rows':len(meta),'layers':414,'readouts':552,'tests_passed':53})
    save_json(out/'artifact_manifest.json',{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['artifact_manifest.json','analysis.log']})
    print(m[m.target.isin(['realized_volatility_5','future_state'])][['representation','target','ic','balanced_accuracy']].to_string(index=False),flush=True)
    print('Report: '+str(out/'report.md'),flush=True)


if __name__=='__main__':main()
