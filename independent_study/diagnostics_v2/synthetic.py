from pathlib import Path
import time
import numpy as np
import pandas as pd
from iclab.neural import encode, uniforms
from .engine import FastSNN
from .probes import fit_probe, predict_probe
from .common import save_json, classification, source_hashes


def make_pairs(pairs, gap, seed=8201):
    rng=np.random.default_rng(seed+gap)
    base=rng.normal(0,.025,(pairs,12,12)).astype(np.float32)
    base[:,:,0:2]=0
    base[:,:,10]=rng.uniform(.25,.55,(pairs,1))
    base[:,:,11]=.1
    first,second=6-gap,6
    base[:,first,0]=rng.uniform(.65,1.,pairs)
    base[:,second,1]=rng.uniform(.65,1.,pairs)
    opposite=base.copy(); opposite[:,[first,second]]=base[:,[second,first]]
    x=np.stack([base,opposite],axis=1).reshape(pairs*2,12,12)
    return x,np.tile([0,1],pairs),np.repeat(np.arange(pairs),2),(first,second)


def encoded_pairs(x, positions, seed):
    ab=encode(x[::2],np.arange(len(x)//2)+100000,"rate",seed)
    opposite=ab.reshape(-1,12,8,24).copy()
    opposite[:,list(positions)]=opposite[:,list(positions[::-1])]
    return np.stack([ab,opposite.reshape(-1,96,24)],axis=1).reshape(len(x),96,24)


def shuffle_blocks(spikes,ids,seed):
    blocks=spikes.reshape(-1,12,8,24)
    order=np.argsort(uniforms(ids,(11,),seed+7919),axis=1)
    order=np.c_[order,np.full(len(order),11)]
    return blocks[np.arange(len(blocks))[:,None],order].reshape(spikes.shape)


def run(cfg,v1):
    out=Path(cfg["output_dir"])/"synthetic"; out.mkdir(parents=True,exist_ok=True)
    if (out/"complete.json").exists(): return
    nt,nv,ne=[cfg["synthetic_pairs"][n] for n in ["train","validation","test"]]
    summary=[]; predictions=[]; started=time.monotonic()
    for gap in cfg["synthetic_gaps_minutes"]:
        x,labels,ids,pos=make_pairs(nt+nv+ne,gap)
        train=ids<nt; val=(ids>=nt)&(ids<nt+nv); test=ids>=nt+nv
        y=np.eye(2)[labels]
        for seed in cfg["seeds"]:
            spikes=encoded_pairs(x,pos,seed)
            assert np.array_equal(spikes[::2].sum(axis=1),spikes[1::2].sum(axis=1))
            assert np.array_equal(spikes[::2,-24:],spikes[1::2,-24:])
            reps={"raw_history":x.reshape(len(x),-1),"counts_total":spikes.sum(axis=1).astype(float),"counts_minute":spikes.reshape(-1,12,8,24).sum(axis=2).reshape(len(x),-1).astype(float)}
            diagnostics={}
            for mode in cfg["modes"]:
                s=shuffle_blocks(spikes,ids,seed) if mode=="shuffle" else spikes
                model=FastSNN(24,v1["network"],seed)
                model.fit(s[train],plastic=mode!="random")
                reps[mode]=model.transform(s)
                model.save(out/f"gap{gap}_{mode}_s{seed}_snn.npz")
                diagnostics[mode]=model.diagnostics(reps[mode][train])
            probe_models={}
            for name,r in reps.items():
                for kind in cfg["probe_types"]:
                    key=f"gap{gap}_{name}_{kind}_s{seed}"
                    model=fit_probe(r[train],r[val],y[train],y[val],kind,cfg["alphas"],components=cfg["rff_components"],groups=[[0,1]],nonnegative=())
                    np.savez_compressed(out/(key+"_probe.npz"),**model)
                    probe_models[key]=(name,kind,model,r)
            save_json(out/f"gap{gap}_s{seed}_lock.json",{"gap":gap,"seed":seed,"train_pairs":nt,"validation_pairs":nv,"future_test_pairs":ne,"sources":source_hashes(),"same_counts_and_tail_verified":True,"diagnostics":diagnostics})
            for key,(name,kind,model,r) in probe_models.items():
                p=predict_probe(model,r[test]); m=classification(labels[test],p,2)
                summary.append({"gap_minutes":gap,"seed":seed,"representation":name,"probe":kind,**{k:v for k,v in m.items() if k not in ["confusion","support"]}})
                predictions.append(pd.DataFrame({"gap":gap,"seed":seed,"representation":name,"probe":kind,"pair_id":ids[test],"actual":labels[test],"prediction":p.argmax(axis=1),"score_0":p[:,0],"score_1":p[:,1]}))
            pd.DataFrame(summary).to_csv(out/"metrics.csv",index=False)
            print(f"synthetic gap={gap} seed={seed} complete",flush=True)
    pd.concat(predictions,ignore_index=True).to_parquet(out/"predictions.parquet",index=False)
    save_json(out/"complete.json",{"seconds":time.monotonic()-started,"models":len(cfg["synthetic_gaps_minutes"])*len(cfg["seeds"])*len(cfg["modes"]),"rows":len(summary)})
