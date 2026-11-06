"""Independent known-answer mechanism check; not a financial tuning sweep."""
import json
import numpy as np
import pandas as pd
from sequence_v8.core import generate,delayed_inputs,output_training_arrays
from event_stdp_v7.core import fit_stdp,ranking
from finance_v8.core import hidden_threshold
from eligibility_v9.core import EventModulatedPatterns,eligibility,auxiliary_delta
from eligibility_v9.train import HERE,ROOT,ARMS
from iclab.data import dump_json,sha256


def main():
    cfg=json.loads((HERE/'config.json').read_text());dest=ROOT/'independent_study/results/v9_mechanism'
    if dest.exists():raise RuntimeError('Refuse to overwrite mechanism')
    dest.mkdir(parents=True);records=[]
    for seed in cfg['seeds']:
        train=generate(200,seed);test=generate(100,seed+10000)
        x=delayed_inputs(train['spikes'],train['anchors'],cfg['delays'])
        xt=delayed_inputs(test['spikes'],test['anchors'],cfg['delays'])
        y=train['target'];fake=np.random.default_rng(seed).permutation(y);p=float(y.mean())
        original=None
        for arm in ARMS:
            feedback=None if arm=='unsupervised' else (y if arm=='true_feedback' else fake)
            m=EventModulatedPatterns(x.shape[1],cfg,seed).fit(x,seed+1,feedback,p)
            if original is None:original=m.weight.copy()
            theta=hidden_threshold(m.weight,x,cfg['hidden_quantile']);h=m.response(x,theta)[0];ht=m.response(xt,theta)[0]
            spikes,event,seg=output_training_arrays(h,train)
            w,stats=fit_stdp(spikes,event,seg,np.ones(len(seg),bool),cfg['output'])
            pred=ht@w;strong=np.isin(test['kinds'],[0,1,2,3,4])
            records.append(dict(seed=seed,model=arm,**ranking(test['target'],pred),
                temporal_subset_auc=ranking(test['target'][strong],pred[strong])['auc'],
                hidden_change=float(abs(m.weight-original).mean()),train_firing=float(h.mean()),**m.audit))
            np.savez_compressed(dest/f'{arm}_{seed}.npz',hidden=m.weight,theta=theta,output=w,target=test['target'],prediction=pred)
    pd.DataFrame(records).to_csv(dest/'metrics.csv',index=False)
    pre=np.array([1.,0.,1.]);w=np.array([.2,.3,.4]);e=eligibility(pre,5,5)
    dump_json(dest/'local_credit.json',dict(weight=w,pre=pre,eligibility=e,
        positive_delta=auxiliary_delta(w,e,.9,.15),negative_delta=auxiliary_delta(w,e,-.1,.15),
        interpretation='Signed auxiliary term before original LTP and normalization; nonparticipants receive zero auxiliary credit.'))
    dump_json(dest/'manifest.json',dict(artifacts={str(p.relative_to(ROOT)):sha256(p) for p in dest.iterdir() if p.is_file()},
        sources={str(p.relative_to(ROOT)):sha256(p) for p in [HERE/'core.py',HERE/'mechanism.py',HERE/'config.json']}))
    print(pd.DataFrame(records)[['seed','model','auc','temporal_subset_auc','hidden_change']].to_string(index=False))


if __name__=='__main__':main()
