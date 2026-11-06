import argparse
import json
import time
import numpy as np
import pandas as pd
from iclab.data import FEATURES,dump_json,sha256
from research_v10.state_core import categories,one_hot,predict_variance
from research_v10.state_run import ROOT,HERE,OUT as STATES,load,metric
from research_v10.modulated_core import softmax
from research_v10.temperature_core import posterior,fit_output

OUT=ROOT/'independent_study/results/v10_temperature'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['develop','audit']);args=parser.parse_args()
    cfg=json.loads((HERE/'config.json').read_text());bars,meta,folds,y,base,current=load()
    k=json.loads((STATES/'selection_lock.json').read_text())['selected_hidden'];features=bars[FEATURES].to_numpy()
    ids=meta.bar_id.to_numpy();seg=bars.segment_id.to_numpy();started=time.time()
    if args.stage=='develop':
        if OUT.exists():raise RuntimeError('Refuse to overwrite temperature experiment')
        OUT.mkdir(parents=True)
        files=[HERE/'temperature_core.py',HERE/'temperature_run.py',HERE/'STAGE_E.md',HERE/'config.json']
        dump_json(OUT/'source_lock.json',{str(p.relative_to(ROOT)):sha256(p) for p in files})
    else:
        assert (ROOT/'independent_study/results/v10_modulated/audit_forecasts.parquet').exists()
        for f,h in json.loads((OUT/'source_lock.json').read_text()).items():assert sha256(ROOT/f)==h,f
    frames=[];diagnostics=[];errors=[]
    for fold in folds:
        fid=fold['fold'];dev=fid in cfg['development_folds']
        if dev!=(args.stage=='develop'):continue
        parent=STATES/f'develop_k{k}' if dev else STATES/'audit';old=parent/f'fold_{fid:02d}'
        fd=OUT/f'fold_{fid:02d}';fd.mkdir(exist_ok=False)
        cat,raw=categories(features,np.load(old/'preprocess.npz')['bounds'],seg,ids,cfg['delays']);pre=one_hot(cat,cfg['bins'])
        baseline=predict_variance(np.load(old/'raw_readout.npz'),np.c_[base,raw])
        train=meta.trading_date.isin(fold['train_dates']).to_numpy();val=meta.trading_date.isin(fold['validation_dates']).to_numpy()
        future=meta.trading_date.isin(fold['forecast_dates']).to_numpy();values={}
        for source,temp,name in [('stdp',1.,'stdp_t1'),('stdp',6.,'stdp_t6'),('random',6.,'random_t6'),('shuffled',6.,'shuffled_t6')]:
            pieces=[]
            for seed in cfg['seeds']:
                q=np.load(old/f'{source}_{seed}.npz')['q'];q=softmax(np.log(q),axis=2)
                p=posterior(q,pre,temp);prediction,model=fit_output(p,baseline,y,train,val,seed+1)
                pieces.append(prediction);np.savez_compressed(fd/f'{name}_{seed}.npz',**model)
                diagnostics.append(dict(fold=fid,model=name,seed=seed,epoch=model['selected_epoch'],
                    train_entropy=float((-p[train]*np.log(np.maximum(p[train],1e-20))).sum(axis=1).mean()),
                    forecast_entropy=float((-p[future]*np.log(np.maximum(p[future],1e-20))).sum(axis=1).mean())))
            values[name]=np.mean(pieces,axis=0)
        values['raw']=baseline
        for split,mask in [('validation',val),('forecast',future)]:
            prior=pd.read_parquet(ROOT/f'independent_study/results/v10_modulated/fold_{fid:02d}/{split}.parquet')
            error=float(abs(prior.frozen_stdp.to_numpy()-values['stdp_t1'][mask]).max());assert error<1e-7,error;errors.append(error)
            frame=meta.loc[mask,['sample_id','bar_id','trading_date','datetime','contract_id']].copy();frame['fold']=fid;frame['realized']=y[mask]
            np.testing.assert_array_equal(frame.bar_id,prior.bar_id)
            for name,p in values.items():frame[name]=p[mask]
            frame['gated_stdp']=prior.gated_stdp.to_numpy();frame['random_t1']=prior.frozen_random.to_numpy()
            frame.to_parquet(fd/f'{split}.parquet',index=False)
            if split=='forecast':frames.append(frame)
        dump_json(fd/'lock.json',dict(fold=fold,files={p.name:sha256(p) for p in fd.iterdir() if p.is_file()}))
        print(f'{args.stage} temperature fold {fid}; {time.time()-started:.1f}s',flush=True)
    frame=pd.concat(frames,ignore_index=True);names=list(values)+['gated_stdp','random_t1']
    summary=pd.DataFrame([dict(model=m,**metric(frame.realized.to_numpy(),frame[m].to_numpy())) for m in names])
    prefix='development' if args.stage=='develop' else 'audit'
    frame.to_parquet(OUT/f'{prefix}_forecasts.parquet',index=False);summary.to_csv(OUT/f'{prefix}_metrics.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/f'{prefix}_diagnostics.csv',index=False)
    dump_json(OUT/f'{prefix}_verification.json',dict(temperature1_max_prediction_difference=max(errors),reproduced_split_groups=len(errors)))
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':main()
