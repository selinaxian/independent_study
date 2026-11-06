"""Explain synthetic quantile ties using v8's already documented 1.3 threshold.

Post-mechanism diagnostic only: reuse hidden weights, never change financial
thresholds, never select this variant by financial performance.
"""
import json
import numpy as np
import pandas as pd
from sequence_v8.core import generate,delayed_inputs,output_training_arrays
from event_stdp_v7.core import fit_stdp,ranking
from eligibility_v9.train import ROOT,HERE,ARMS
from iclab.data import dump_json,sha256


def main():
    cfg=json.loads((HERE/'config.json').read_text());src=ROOT/'independent_study/results/v9_mechanism'
    dest=src/'threshold_audit';dest.mkdir(exist_ok=False);rows=[]
    for seed in cfg['seeds']:
        train=generate(200,seed);test=generate(100,seed+10000)
        x=delayed_inputs(train['spikes'],train['anchors'],cfg['delays'])
        xt=delayed_inputs(test['spikes'],test['anchors'],cfg['delays'])
        for arm in ARMS:
            saved=np.load(src/f'{arm}_{seed}.npz');w=saved['hidden'];v=x@w.T
            h=(v>cfg['threshold']).astype(float);ht=(xt@w.T>cfg['threshold']).astype(float)
            s,e,seg=output_training_arrays(h,train)
            out,stats=fit_stdp(s,e,seg,np.ones(len(seg),bool),cfg['output'])
            pred=ht@out;strong=np.isin(test['kinds'],[0,1,2,3,4])
            rows.append(dict(seed=seed,model=arm,threshold=cfg['threshold'],
                quantile_equals_max=int(np.isclose(saved['theta'],v.max(axis=0),atol=1e-12).sum()),
                quantile_silent_units=int(((v>saved['theta']).sum(axis=0)==0).sum()),
                fixed_train_firing=float(h.mean()),**ranking(test['target'],pred),
                temporal_subset_auc=ranking(test['target'][strong],pred[strong])['auc']))
            np.savez_compressed(dest/f'{arm}_{seed}.npz',output=out,prediction=pred,target=test['target'])
    pd.DataFrame(rows).to_csv(dest/'metrics.csv',index=False)
    dump_json(dest/'manifest.json',dict(artifacts={str(p.relative_to(ROOT)):sha256(p) for p in dest.iterdir() if p.is_file()},
        sources={str((HERE/'mechanism_audit.py').relative_to(ROOT)):sha256(HERE/'mechanism_audit.py')},
        note='Exploratory explanation of sparse synthetic threshold ties, not a new financial variant.'))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
