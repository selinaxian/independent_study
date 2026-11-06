import argparse
import json
import time
import numpy as np
import pandas as pd
from iclab.data import FEATURES,dump_json,sha256
from eligibility_v9.core import mismatch_feedback
from research_v10.modulated_core import fit_modulated,forward
from research_v10.state_core import categories,one_hot,predict_variance,qlike
from research_v10.state_run import ROOT,HERE,OUT as STATES,load,metric

OUT=ROOT/'independent_study/results/v10_modulated'
MODELS=['frozen_stdp','true_modulated','mismatched','random_modulated','frozen_random']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['develop','audit']);args=parser.parse_args()
    cfg=json.loads((HERE/'config.json').read_text());bars,meta,folds,y,base,current=load()
    k=json.loads((STATES/'selection_lock.json').read_text())['selected_hidden'];features=bars[FEATURES].to_numpy()
    seg=bars.segment_id.to_numpy();ids=meta.bar_id.to_numpy();started=time.time()
    if args.stage=='develop':
        if OUT.exists():raise RuntimeError('Refuse to overwrite modulation')
        OUT.mkdir(parents=True)
        paths=[HERE/'modulated_core.py',HERE/'modulated_run.py',HERE/'STAGE_D.md',HERE/'config.json',
               HERE/'state_core.py',ROOT/'independent_study/eligibility_v9/core.py']
        dump_json(OUT/'source_lock.json',{str(p.relative_to(ROOT)):sha256(p) for p in paths})
    else:
        assert (OUT/'development_metrics.csv').exists()
        for path,h in json.loads((OUT/'source_lock.json').read_text()).items():assert sha256(ROOT/path)==h,path
    frames=[];diags=[]
    for fold in folds:
        fid=fold['fold'];dev=fid in cfg['development_folds']
        if dev!=(args.stage=='develop'):continue
        parent=STATES/f'develop_k{k}' if dev else STATES/'audit';old=parent/f'fold_{fid:02d}'
        fd=OUT/f'fold_{fid:02d}';fd.mkdir(exist_ok=False)
        bounds=np.load(old/'preprocess.npz')['bounds'];cat,raw=categories(features,bounds,seg,ids,cfg['delays']);pre=one_hot(cat,cfg['bins'])
        baseline=predict_variance(np.load(old/'raw_readout.npz'),np.c_[base,raw])
        train=meta.trading_date.isin(fold['train_dates']).to_numpy();val=meta.trading_date.isin(fold['validation_dates']).to_numpy()
        forecast=meta.trading_date.isin(fold['forecast_dates']).to_numpy()
        pieces={m:[] for m in MODELS}
        for seed in cfg['seeds']:
            wrong=mismatch_feedback(y,meta.segment_id.to_numpy(),train,seed+3)
            for name in MODELS:
                source='random' if name in ['random_modulated','frozen_random'] else 'stdp'
                initial=np.load(old/f'{source}_{seed}.npz')['q']
                mode='frozen' if name.startswith('frozen') else 'mismatched' if name=='mismatched' else 'true'
                pred,model=fit_modulated(pre,baseline,y,train,val,initial,mode,seed+1,wrong)
                pieces[name].append(pred);np.savez_compressed(fd/f'{name}_{seed}.npz',**model)
                diags.append(dict(fold=fid,seed=seed,model=name,epoch=model['selected_epoch'],
                    validation_qlike=model['validation_qlike'],q_change=model['q_change'],
                    a_norm=float(np.linalg.norm(model['a'])),bias=float(model['b'])))
        predictions={m:np.mean(v,axis=0) for m,v in pieces.items()};predictions['raw']=baseline
        for split,mask in [('validation',val),('forecast',forecast)]:
            frame=meta.loc[mask,['sample_id','bar_id','trading_date','datetime','contract_id']].copy()
            frame['fold']=fid;frame['realized']=y[mask]
            for name,p in predictions.items():frame[name]=p[mask]
            prior=pd.read_parquet(ROOT/f'independent_study/results/v10_increment/fold_{fid:02d}/{split}.parquet')
            np.testing.assert_array_equal(frame.bar_id,prior.bar_id)
            frame['gated_stdp']=prior.gated_stdp.to_numpy()
            frame.to_parquet(fd/f'{split}.parquet',index=False)
            if split=='forecast':frames.append(frame)
        dump_json(fd/'lock.json',dict(fold=fold,files={p.name:sha256(p) for p in fd.iterdir() if p.is_file()}))
        print(f'{args.stage} modulated fold {fid}; {time.time()-started:.1f}s',flush=True)
    data=pd.concat(frames,ignore_index=True)
    names=MODELS+['raw','gated_stdp']
    summary=pd.DataFrame([dict(model=m,**metric(data.realized.to_numpy(),data[m].to_numpy())) for m in names])
    prefix='development' if args.stage=='develop' else 'audit'
    summary.to_csv(OUT/f'{prefix}_metrics.csv',index=False);data.to_parquet(OUT/f'{prefix}_forecasts.parquet',index=False)
    pd.DataFrame(diags).to_csv(OUT/f'{prefix}_diagnostics.csv',index=False)
    for path,h in json.loads((OUT/'source_lock.json').read_text()).items():assert sha256(ROOT/path)==h,path
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':main()
