"""Matched AB/BA known-order check; independent of financial outcomes."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from diagnostics_v2.synthetic import make_pairs
from diagnostics_v2.common import config,save_json,classification
from diagnostics_v2.engine import FastSNN
from diagnostics_v2.probes import fit_probe,predict_probe
from .run import settings,sources
from .encoding import latency,time_codes


def run():
    cfg=settings();old,v1=config();out=Path(cfg['output'])/'synthetic';out.mkdir(exist_ok=True)
    torch.set_num_threads(1);nt,nv,ne=[cfg['synthetic_pairs'][key] for key in ['train','validation','test']]
    metrics=[];predictions=[];start=time.monotonic()
    for gap in cfg['synthetic_gaps_minutes']:
        x,labels,ids,pos=make_pairs(nt+nv+ne,gap);s=latency(x);y=np.eye(2)[labels]
        np.testing.assert_array_equal(s[::2].sum(axis=1),s[1::2].sum(axis=1))
        np.testing.assert_array_equal(s[::2,-24:],s[1::2,-24:])
        train=ids<nt;val=(ids>=nt)&(ids<nt+nv);test=ids>=nt+nv
        for seed in cfg['seeds']:
            reps={'raw_history':x.reshape(len(x),-1),'time_code':time_codes(s).reshape(len(x),-1),
                  'counts_total':s.sum(axis=1).astype(float),'counts_minute':s.reshape(-1,12,8,24).sum(axis=2).reshape(len(x),-1).astype(float)}
            for mode in cfg['modes']:
                model=FastSNN(24,v1['network'],seed);model.fit(s[train],plastic=mode=='stdp')
                model.save(out/f'gap{gap}_{mode}_s{seed}_snn.npz');reps[mode]=model.transform(s)
            models={}
            for name,r in reps.items():
                model=fit_probe(r[train],r[val],y[train],y[val],'linear',old['alphas'],groups=[[0,1]],nonnegative=())
                np.savez_compressed(out/f'gap{gap}_{name}_s{seed}_probe.npz',**model);models[name]=model
            save_json(out/f'gap{gap}_s{seed}_lock.json',{'train_pairs':nt,'validation_pairs':nv,'test_pairs':ne,
                  'same_counts_same_recent_tail':True,'sources':sources()})
            for name,model in models.items():
                p=predict_probe(model,reps[name][test]);m=classification(labels[test],p,2)
                metrics.append({'gap_minutes':gap,'seed':seed,'representation':name,'probe':'linear',
                    **{k:v for k,v in m.items() if k not in ['confusion','support']}})
                predictions.append(pd.DataFrame({'gap':gap,'seed':seed,'representation':name,'pair_id':ids[test],
                     'actual':labels[test],'prediction':p.argmax(axis=1),'score_0':p[:,0],'score_1':p[:,1]}))
            print(f'latency synthetic gap={gap} seed={seed} complete',flush=True)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(out/'predictions.parquet',index=False)
    save_json(out/'complete.json',{'seconds':time.monotonic()-start,'models':12,'readouts':len(metrics)})


if __name__=='__main__':run()
