import json
import numpy as np
import pandas as pd
from iclab.data import FEATURES,dump_json,sha256
from research_v10.state_core import (ProbabilitySTDP,categories,fit_bins,one_hot,fit_variance,
                                     predict_variance,qlike,shuffle_delays)
from research_v10.state_run import ROOT,HERE,OUT,load,metric

NAMES=['current','har','raw','encoded','stdp','random','shuffled','state_only']


def paired_blocks(data,cfg,scope):
    days=data.trading_date.unique();counts=np.array([(data.trading_date==d).sum() for d in days])
    y=data.realized.to_numpy();loss={m:qlike(y,data[m].to_numpy()) for m in NAMES}
    dayloss={m:np.array([loss[m][data.trading_date.to_numpy()==d].sum() for d in days]) for m in NAMES}
    rng=np.random.default_rng(cfg['seed']);weight=[];block=cfg['bootstrap_block_days']
    for _ in range(cfg['bootstrap_repeats']):
        starts=rng.integers(0,len(days)-block+1,int(np.ceil(len(days)/block)))
        idx=np.concatenate([np.arange(s,s+block) for s in starts])[:len(days)]
        weight.append(np.bincount(idx,minlength=len(days)))
    weight=np.asarray(weight);denominator=weight@counts;result=[]
    for other in NAMES:
        if other=='stdp':continue
        vals=(weight@(dayloss['stdp']-dayloss[other]))/denominator
        lo,hi=np.quantile(vals,[.025,.975])
        result.append(dict(scope=scope,other=other,metric='QLIKE lower is better',
            difference=float((loss['stdp']-loss[other]).mean()),lower=lo,upper=hi))
    return result


def main():
    cfg=json.loads((HERE/'config.json').read_text());selection=json.loads((OUT/'selection_lock.json').read_text());k=selection['selected_hidden']
    assert json.loads((OUT/'stage_b_status.json').read_text())['status']=='complete'
    bars,meta,folds,y,base,current=load();ids=meta.bar_id.to_numpy();features=bars[FEATURES].to_numpy();seg=bars.segment_id.to_numpy()
    dev=pd.read_parquet(OUT/f'develop_k{k}/forecasts.parquet');audit=pd.read_parquet(OUT/'audit/forecasts.parquet')
    combined=pd.concat([dev,audit],ignore_index=True).sort_values('bar_id');assert not combined.bar_id.duplicated().any()
    combined.to_parquet(OUT/'forecasts.parquet',index=False)
    summaries=[];intervals=[]
    for scope,data in [('development',dev),('audit',audit),('all_exploratory',combined)]:
        summaries.extend(dict(scope=scope,model=m,**metric(data.realized.to_numpy(),data[m].to_numpy())) for m in NAMES)
        intervals.extend(paired_blocks(data,cfg,scope))
    pd.DataFrame(summaries).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(intervals).to_csv(OUT/'paired_intervals.csv',index=False)
    bymonth=combined.assign(month=combined.trading_date.str[:7])
    monthly=[dict(month=month,model=m,**metric(g.realized.to_numpy(),g[m].to_numpy())) for month,g in bymonth.groupby('month') for m in NAMES]
    pd.DataFrame(monthly).to_csv(OUT/'monthly_metrics.csv',index=False)
    errors=[];retrained=0;diags=[];profiles=[];seedframes=[]
    for fold in folds:
        fid=fold['fold'];parent=OUT/f'develop_k{k}' if fid in cfg['development_folds'] else OUT/'audit';fd=parent/f'fold_{fid:02d}'
        lock=json.loads((fd/'lock.json').read_text())
        for path,expected in lock['files'].items():assert sha256(fd/path)==expected,path
        p=np.load(fd/'preprocess.npz');tb=bars.trading_date.isin(fold['train_dates']).to_numpy()&np.isfinite(features).all(axis=1)
        np.testing.assert_array_equal(p['bounds'],fit_bins(features,tb,cfg['bins']))
        assert max(fold['train_dates'])<min(fold['validation_dates']) and max(fold['validation_dates'])<min(fold['forecast_dates'])
        diags.append(pd.read_csv(fd/'diagnostics.csv'));profiles.append(pd.read_csv(fd/'state_profiles.csv'));seedframes.append(pd.read_parquet(fd/'seed_forecasts.parquet'))
        if fid not in [0,11,22]:continue
        cat,raw=categories(features,p['bounds'],seg,ids,cfg['delays']);hot=one_hot(cat,cfg['bins'])
        train=meta.trading_date.isin(fold['train_dates']).to_numpy();val=meta.trading_date.isin(fold['validation_dates']).to_numpy()
        pred={'current':current}
        for name,x in [('har',base),('raw',np.c_[base,raw]),('encoded',np.c_[base,hot])]:pred[name]=predict_variance(np.load(fd/f'{name}_readout.npz'),x)
        pieces={m:[] for m in ['stdp','random','shuffled','state_only']}
        for seed in cfg['seeds']:
            for kind in ['stdp','random','shuffled']:
                saved=np.load(fd/f'{kind}_{seed}.npz')
                net=ProbabilitySTDP(cat.shape[1],cfg['bins'],k,cfg['eta'],cfg['epochs'],seed,cfg['probability_floor'])
                if seed==104 and kind in ['stdp','shuffled']:
                    traincat=cat[train] if kind=='stdp' else shuffle_delays(cat[train],len(FEATURES),len(cfg['delays']),seed+2)
                    net.fit(traincat,seed+1);np.testing.assert_array_equal(net.q,saved['q']);retrained+=1
                net.q=saved['q'];post,_=net.posterior(cat)
                model={key.removeprefix('readout_'):saved[key] for key in saved.files if key.startswith('readout_')}
                pieces[kind].append(predict_variance(model,np.c_[base,post]))
                if kind=='stdp':pieces['state_only'].append(predict_variance(np.load(fd/f'state_only_{seed}.npz'),post))
        for name,values in pieces.items():pred[name]=np.mean(values,axis=0)
        for split in ['validation','forecast']:
            frame=pd.read_parquet(fd/f'{split}.parquet');index=frame.sample_id.to_numpy()
            np.testing.assert_array_equal(meta.sample_id,np.arange(len(meta)))
            for name,values in pred.items():
                err=float(abs(frame[name]-values[index]).max());assert err<1e-7,(fid,name,err);errors.append(err)
    for name,frames in [('diagnostics',diags),('state_profiles',profiles)]:pd.concat(frames,ignore_index=True).to_csv(OUT/f'{name}.csv',index=False)
    pd.concat(seedframes,ignore_index=True).to_parquet(OUT/'seed_forecasts.parquet',index=False)
    for path,expected in json.loads((OUT/'source_lock.json').read_text()).items():assert sha256(ROOT/path)==expected,path
    for path,expected in selection['development_files'].items():assert sha256(OUT/path)==expected,path
    dump_json(OUT/'verification.json',dict(status='complete',forecasts=len(combined),train_only_bins_all_folds=True,
        exact_hidden_retraining=retrained,reloaded_prediction_groups=len(errors),max_prediction_error=max(errors),selection_lock_unchanged=True,
        executed_window_target_recomputed=True,development_and_audit_separate=True))
    print(pd.DataFrame(summaries).to_string(index=False));print(pd.DataFrame(intervals).to_string(index=False))


if __name__=='__main__':main()
