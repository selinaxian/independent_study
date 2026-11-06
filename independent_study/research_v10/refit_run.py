import argparse
import json
import time
import numpy as np
import pandas as pd
from iclab.data import FEATURES,dump_json,sha256
from research_v10.state_core import ProbabilitySTDP,fit_bins,categories,one_hot,shuffle_delays
from research_v10.temperature_core import posterior
from research_v10.refit_core import fixed_output,fixed_variance
from research_v10.state_run import ROOT,HERE,OUT as STATES,load,metric

OUT=ROOT/'independent_study/results/v10_refit'
TEMPERATURE=ROOT/'independent_study/results/v10_temperature'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['develop','audit']);args=parser.parse_args()
    cfg=json.loads((HERE/'config.json').read_text());bars,meta,folds,y,base,current=load();ids=meta.bar_id.to_numpy()
    k=json.loads((STATES/'selection_lock.json').read_text())['selected_hidden'];features=bars[FEATURES].to_numpy();seg=bars.segment_id.to_numpy()
    if args.stage=='develop':
        if OUT.exists():raise RuntimeError('Refuse to overwrite refit')
        OUT.mkdir(parents=True)
        files=[HERE/'refit_core.py',HERE/'refit_run.py',HERE/'STAGE_F.md',HERE/'config.json',HERE/'state_core.py',HERE/'temperature_core.py']
        dump_json(OUT/'source_lock.json',{str(p.relative_to(ROOT)):sha256(p) for p in files})
    else:
        assert (OUT/'development_metrics.csv').exists()
        for f,h in json.loads((OUT/'source_lock.json').read_text()).items():assert sha256(ROOT/f)==h,f
    frames=[];diagnostics=[];started=time.time()
    for fold in folds:
        fid=fold['fold'];dev=fid in cfg['development_folds']
        if dev!=(args.stage=='develop'):continue
        old=(STATES/f'develop_k{k}' if dev else STATES/'audit')/f'fold_{fid:02d}'
        reference=TEMPERATURE/f'fold_{fid:02d}';fd=OUT/f'fold_{fid:02d}';fd.mkdir(exist_ok=False)
        past_dates=fold['train_dates']+fold['validation_dates'];assert max(past_dates)<min(fold['forecast_dates'])
        past=meta.trading_date.isin(past_dates).to_numpy();future=meta.trading_date.isin(fold['forecast_dates']).to_numpy()
        bpast=bars.trading_date.isin(past_dates).to_numpy()&np.isfinite(features).all(axis=1)
        bounds=fit_bins(features,bpast,cfg['bins']);cat,raw=categories(features,bounds,seg,ids,cfg['delays'])
        oldcat,_=categories(features,np.load(old/'preprocess.npz')['bounds'],seg,ids,cfg['delays'])
        hot=one_hot(cat,cfg['bins']);oldhot=one_hot(oldcat,cfg['bins'])
        oldbase=np.load(old/'raw_readout.npz');baseline,bmodel=fixed_variance(np.c_[base,raw],y,past,float(oldbase['alpha']))
        np.savez_compressed(fd/'raw_readout.npz',**bmodel);np.savez_compressed(fd/'preprocess.npz',bounds=bounds)
        predictions={'raw50':baseline}
        for source in ['stdp','random','shuffled']:
            pieces={schedule:[] for schedule in ['readout50','full50']}
            for seed in cfg['seeds']:
                q_old=np.load(old/f'{source}_{seed}.npz')['q']
                epochs=int(np.load(reference/f'{source}_t6_{seed}.npz')['selected_epoch'])
                net=ProbabilitySTDP(cat.shape[1],cfg['bins'],k,cfg['eta'],cfg['epochs'],seed,cfg['probability_floor'])
                if source!='random':
                    inputs=cat[past] if source=='stdp' else shuffle_delays(cat[past],len(FEATURES),len(cfg['delays']),seed+2)
                    net.fit(inputs,seed+1)
                for schedule,q,x in [('readout50',q_old,oldhot),('full50',net.q,hot)]:
                    p=posterior(q,x,6.);pred,model=fixed_output(p,baseline,y,past,epochs,seed+1)
                    pieces[schedule].append(pred)
                    np.savez_compressed(fd/f'{schedule}_{source}_{seed}.npz',q=q,**model)
                    diagnostics.append(dict(fold=fid,model=source,schedule=schedule,seed=seed,epochs=epochs,
                        training_samples=int(past.sum()),forecast_entropy=float((-p[future]*np.log(np.maximum(p[future],1e-20))).sum(axis=1).mean())))
            for schedule,values in pieces.items():predictions[f'{schedule}_{source}']=np.mean(values,axis=0)
        frame=meta.loc[future,['sample_id','bar_id','trading_date','datetime','contract_id']].copy();frame['fold']=fid;frame['realized']=y[future]
        prior=pd.read_parquet(reference/'forecast.parquet');np.testing.assert_array_equal(frame.bar_id,prior.bar_id)
        for name,values in predictions.items():frame[name]=values[future]
        frame['raw40']=prior.raw.to_numpy();frame['old40_stdp']=prior.stdp_t6.to_numpy()
        frame['old40_random']=prior.random_t6.to_numpy();frame['gated40_stdp']=prior.gated_stdp.to_numpy()
        frame.to_parquet(fd/'forecast.parquet',index=False);frames.append(frame)
        dump_json(fd/'lock.json',dict(fold=fold,past_dates=past_dates,selected_alpha=float(oldbase['alpha']),
            epochs_source=str(reference.relative_to(ROOT)),files={p.name:sha256(p) for p in fd.iterdir() if p.is_file()}))
        print(f'{args.stage} refit fold {fid}; {time.time()-started:.1f}s',flush=True)
    frame=pd.concat(frames,ignore_index=True);names=list(predictions)+['raw40','old40_stdp','old40_random','gated40_stdp']
    summary=pd.DataFrame([dict(model=m,**metric(frame.realized.to_numpy(),frame[m].to_numpy())) for m in names])
    prefix='development' if args.stage=='develop' else 'audit'
    frame.to_parquet(OUT/f'{prefix}_forecasts.parquet',index=False);summary.to_csv(OUT/f'{prefix}_metrics.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/f'{prefix}_diagnostics.csv',index=False)
    for f,h in json.loads((OUT/'source_lock.json').read_text()).items():assert sha256(ROOT/f)==h,f
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':main()
