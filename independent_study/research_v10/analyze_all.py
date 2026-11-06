"""Read-only analysis of frozen stages; no model/threshold selection here."""
import json
import numpy as np
import pandas as pd
from iclab.data import FEATURES,dump_json
from research_v10.state_run import ROOT,HERE,load,metric
from research_v10.state_core import categories,one_hot,predict_variance,qlike
from research_v10.temperature_core import posterior,predict
from risk_v4.core import block_interval

OUT=ROOT/'independent_study/results/v10'
RESULTS=OUT.parent
PAIRS={
 'modulated':[('true_modulated',m) for m in ['raw','frozen_stdp','mismatched','random_modulated','frozen_random']],
 'temperature':[('stdp_t6',m) for m in ['raw','stdp_t1','random_t6','shuffled_t6']],
 'refit':[('readout50_stdp',m) for m in ['raw50','readout50_random','readout50_shuffled','full50_stdp','old40_stdp','gated40_stdp']]+[('raw50','raw40')]
}


def paired(frame,pairs,cfg,scope):
    days=frame.trading_date.unique();counts=np.array([(frame.trading_date==d).sum() for d in days]);rng=np.random.default_rng(cfg['seed'])
    weights=[];block=cfg['bootstrap_block_days']
    for _ in range(cfg['bootstrap_repeats']):
        starts=rng.integers(0,len(days)-block+1,int(np.ceil(len(days)/block)))
        ix=np.concatenate([np.arange(s,s+block) for s in starts])[:len(days)]
        weights.append(np.bincount(ix,minlength=len(days)))
    weights=np.array(weights);denom=weights@counts;rows=[]
    for left,right in pairs:
        delta=qlike(frame.realized.to_numpy(),frame[left].to_numpy())-qlike(frame.realized.to_numpy(),frame[right].to_numpy())
        sums=np.array([delta[frame.trading_date.to_numpy()==d].sum() for d in days]);dist=weights@sums/denom
        lo,hi=np.quantile(dist,[.025,.975]);rows.append(dict(scope=scope,left=left,right=right,difference=float(delta.mean()),lower=lo,upper=hi))
    return rows


def main():
    OUT.mkdir(exist_ok=True);cfg=json.loads((HERE/'config.json').read_text());metrics=[];intervals=[];months=[];foldmetrics=[]
    for stage,pairs in PAIRS.items():
        directory=RESULTS/f'v10_{stage}'
        frame=pd.concat([pd.read_parquet(directory/f'{s}_forecasts.parquet') for s in ['development','audit']],ignore_index=True).sort_values('bar_id')
        assert len(frame)==15900 and not frame.bar_id.duplicated().any()
        frame.to_parquet(directory/'forecasts.parquet',index=False)
        names=[c for c in frame.columns if c not in ['sample_id','bar_id','trading_date','datetime','contract_id','fold','realized']]
        for scope,g in [('development',frame[frame.fold.isin(cfg['development_folds'])]),('audit',frame[~frame.fold.isin(cfg['development_folds'])]),('all_exploratory',frame)]:
            metrics.extend(dict(stage=stage,scope=scope,model=m,**metric(g.realized.to_numpy(),g[m].to_numpy())) for m in names)
            intervals.extend(dict(stage=stage,**r) for r in paired(g,pairs,cfg,scope))
        for month,g in frame.assign(month=frame.trading_date.str[:7]).groupby('month'):
            months.extend(dict(stage=stage,month=month,model=m,**metric(g.realized.to_numpy(),g[m].to_numpy())) for m in names)
        for fold,g in frame.groupby('fold'):
            foldmetrics.extend(dict(stage=stage,fold=fold,model=m,**metric(g.realized.to_numpy(),g[m].to_numpy())) for m in names)
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(intervals).to_csv(OUT/'paired_intervals.csv',index=False)
    pd.DataFrame(months).to_csv(OUT/'monthly_metrics.csv',index=False);pd.DataFrame(foldmetrics).to_csv(OUT/'fold_metrics.csv',index=False)
    # Calibration bins describe held-forward predictions; they do not set any trading rules.
    frame=pd.read_parquet(RESULTS/'v10_refit/forecasts.parquet');rows=[]
    for scope,g in [('development',frame[frame.fold<9]),('audit',frame[frame.fold>=9])]:
        for m in ['raw40','raw50','readout50_stdp','full50_random']:
            bucket=pd.qcut(g[m].rank(method='first'),10,labels=False)
            for b,h in g.groupby(bucket):rows.append(dict(scope=scope,model=m,decile=int(b)+1,n=len(h),predicted_variance=h[m].mean(),realized_variance=h.realized.mean()))
    pd.DataFrame(rows).to_csv(OUT/'calibration.csv',index=False)
    # First audit fold, one fixed seed. State IDs are local, not persistent labels across refits.
    bars,meta,folds,y,base,current=load();fold=folds[9];ids=meta.bar_id.to_numpy();features=bars[FEATURES].to_numpy()
    fd=RESULTS/'v10_refit/fold_09';old=RESULTS/'v10_state_risk/audit/fold_09'
    cat,raw=categories(features,np.load(old/'preprocess.npz')['bounds'],bars.segment_id.to_numpy(),ids,cfg['delays'])
    saved=np.load(fd/'readout50_stdp_104.npz');p=posterior(saved['q'],one_hot(cat,cfg['bins']),6.)
    baseline=predict_variance(np.load(fd/'raw_readout.npz'),np.c_[base,raw]);prediction=predict(saved,p,baseline)
    past=meta.trading_date.isin(fold['train_dates']+fold['validation_dates']).to_numpy();future=meta.trading_date.isin(fold['forecast_dates']).to_numpy()
    masses=p[past].sum(axis=0);chosen=np.argsort(-masses)[:3];states=[];fp=[]
    for j in range(p.shape[1]):
        row=dict(unit=j,illustration_selected=bool(j in chosen),output_weight=float(saved['a'][j]),output_bias=float(saved['b']))
        for name,mask in [('past50',past),('forecast',future)]:
            w=p[mask,j];row[f'{name}_mass_fraction']=float(w.mean());row[f'{name}_realized_to_base']=float(w@(y[mask]/baseline[mask])/w.sum())
            row[f'{name}_rv_bp']=float(w@np.sqrt(y[mask])/w.sum())
        states.append(row)
        for f,feature in enumerate(FEATURES):
            for lag,d in enumerate(cfg['delays']):
                gi=f*len(cfg['delays'])+lag
                fp.append(dict(unit=j,feature=feature,delay=d,preferred_bin=float(saved['q'][j,gi]@np.arange(1,6)),
                    past_soft_mean_bin=float(p[past,j]@(cat[past,gi]+1)/p[past,j].sum()),
                    past_unconditional_mean_bin=float((cat[past,gi]+1).mean())))
    pd.DataFrame(states).to_csv(OUT/'illustrative_states.csv',index=False);pd.DataFrame(fp).to_csv(OUT/'illustrative_state_features.csv',index=False)
    dump_json(OUT/'state_interpretation.json',dict(fold=9,seed=104,selected_by='largest past50 soft mass',units=chosen.tolist(),
        forecast_dates=fold['forecast_dates'],forecast_entropy=float((-p[future]*np.log(p[future])).sum(axis=1).mean()),
        forecast_correction_quantiles=np.quantile(prediction[future]/baseline[future],[0,.1,.5,.9,1]).tolist(),
        note='Posterior temperature 6. State labels are per-fold; association is not causal direction or evidence of unique timing value.'))
    # Economic splits use exactly the pre-existing allocations and strict matched control.
    erows=[]
    for path in sorted((RESULTS/'v10_economics').glob('*_opportunities.csv')):
        scenario,model=path.name.removesuffix('_opportunities.csv').split('__');tr=pd.read_csv(path)
        for scope,mask in [('development',tr.fold<9),('audit',tr.fold>=9),('all_exploratory',tr.fold>=0)]:
            g=tr[mask]
            calendar=[d for f in folds if scope=='all_exploratory' or ((f['fold']<9)==(scope=='development')) for d in f['forecast_dates']]
            days=g.groupby('trading_date')[['net_pnl','matched_net_pnl','actual_risk_squared','matched_risk_squared']].sum().reindex(calendar,fill_value=0)
            delta=days.actual_risk_squared-days.matched_risk_squared;lo,hi=block_interval(delta.to_numpy(),1000,5,cfg['seed'])
            nd=days.net_pnl-days.matched_net_pnl;nl,nh=block_interval(nd.to_numpy(),1000,5,cfg['seed'])
            erows.append(dict(scope=scope,scenario=scenario,model=model,opportunities=len(g),trades=int((g.lots>0).sum()),lots=int(g.lots.sum()),
                gross_pnl=float(g.gross_pnl.sum()),net_pnl=float(g.net_pnl.sum()),
                risk_saving_fraction=1-g.actual_risk_squared.sum()/g.matched_risk_squared.sum(),risk_delta_low=lo,risk_delta_high=hi,
                selection_excess=float(nd.sum()),selection_low=nl,selection_high=nh))
    pd.DataFrame(erows).to_csv(OUT/'economic_scopes.csv',index=False)
    selected=pd.DataFrame(intervals);print(selected.query('stage == "refit" and scope == "audit"').to_string(index=False))
    print(pd.DataFrame(erows).query('scenario == "base" and model in ["raw50","readout50_stdp"]') .to_string(index=False))


if __name__=='__main__':main()
