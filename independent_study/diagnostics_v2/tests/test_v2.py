import numpy as np
import pandas as pd
import torch
from iclab.neural import CompetitiveSNN
from diagnostics_v2.engine import FastSNN
from diagnostics_v2.data import make_folds,fit_scale,path_stats,states
from diagnostics_v2.common import config, classification
from diagnostics_v2.probes import fit_probe,predict_probe
from diagnostics_v2.synthetic import make_pairs,encoded_pairs


def test_scripted_matches_reference_training_and_inference(tmp_path):
    _,v1=config(); c=v1["network"].copy()
    torch.set_num_threads(1)
    a,b=CompetitiveSNN(24,c,104),FastSNN(24,c,104)
    spikes=np.random.default_rng(800).random((13,96,24))<.15
    for _ in range(4):
        ra,rb=a.batch(spikes,True,True),b.batch(spikes,True,True)
        np.testing.assert_allclose(ra,rb,rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(a.weight,b.weight,rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(a.theta,b.theta,rtol=1e-6,atol=1e-6)
    w,t=b.weight.clone(),b.theta.clone()
    np.testing.assert_allclose(a.batch(spikes),b.batch(spikes),rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(w,b.weight,rtol=0,atol=0); torch.testing.assert_close(t,b.theta,rtol=0,atol=0)
    b.save(tmp_path/"checkpoint.npz")
    np.testing.assert_array_equal(b.batch(spikes),FastSNN(24,c,104).load(tmp_path/"checkpoint.npz").batch(spikes))


def test_random_weights_not_updated_by_script():
    _,v1=config(); c=v1["network"]
    model=FastSNN(24,c,104)
    s=np.random.default_rng(1).random((8,96,24))<.2
    model.batch(s,True,False)
    torch.testing.assert_close(model.weight,model.initial_weight,rtol=0,atol=0)


def test_folds_only_past_and_do_not_skip_missing_forecast_dates():
    cfg,_=config(); dates=pd.bdate_range("2025-12-01","2026-09-15").strftime("%Y-%m-%d").tolist()
    available=[d for d in dates if d not in ["2026-07-22","2026-07-23"]]
    folds=make_folds(dates,available,cfg)
    assert sum((f["forecast_dates"] for f in folds),[])==[d for d in dates if cfg["first_forecast"]<=d<=cfg["last_forecast"]]
    for f in folds:
        assert len(f["train_dates"])==40 and len(f["validation_dates"])==10
        assert max(f["train_dates"])<min(f["validation_dates"])<max(f["validation_dates"])<min(f["forecast_dates"])
        assert not set(f["train_dates"])&set(f["forecast_dates"])


def test_synthetic_count_and_tail_controls():
    for gap in [1,5]:
        x,y,ids,pos=make_pairs(30,gap)
        s=encoded_pairs(x,pos,104)
        np.testing.assert_array_equal(s[::2].sum(axis=1),s[1::2].sum(axis=1))
        np.testing.assert_array_equal(s[::2,-24:],s[1::2,-24:])
        assert np.any(s[::2]!=s[1::2]) and np.array_equal(y,np.tile([0,1],30))
        np.testing.assert_allclose(x[::2].sum(axis=1),x[1::2].sum(axis=1),rtol=0,atol=1e-7)


def test_state_paths_known_values():
    p=np.array([[100,101,102,103,104,105],[100,101,100,101,100,100.1]])
    rv,e=path_stats(p)
    assert e[0]==1 and e[1]<.1
    np.testing.assert_array_equal(states(rv,e,[1,.5]),[3,2])


def test_probe_train_only_transforms_reload_and_nonlinearity(tmp_path):
    rng=np.random.default_rng(20); x=rng.uniform(-2,2,(300,2)); v=rng.uniform(-2,2,(120,2))
    y=(x[:,0]*x[:,1]>0).astype(int); vy=(v[:,0]*v[:,1]>0).astype(int)
    out={}
    for kind in ["linear","rff"]:
        m=fit_probe(x,v,np.eye(2)[y],np.eye(2)[vy],kind,[.1,1,10],components=128,groups=[[0,1]],nonnegative=())
        np.testing.assert_allclose(m["input_mean"],x.mean(axis=0))
        p=predict_probe(m,v); out[kind]=classification(vy,p,2)["accuracy"]
        np.savez(tmp_path/(kind+".npz"),**m)
        np.testing.assert_array_equal(p,predict_probe(np.load(tmp_path/(kind+".npz")),v))
    assert out["rff"]>.9 and out["rff"]>out["linear"]+.2


def test_scaled_data_and_state_definition_are_past_only():
    from pathlib import Path
    import json
    cfg,_=config(); out=Path(cfg["output_dir"])
    if not (out/"prepared.json").exists(): return
    folds=json.loads((out/"folds.json").read_text()); meta=pd.read_parquet(out/"samples.parquet")
    bars=pd.read_parquet(Path(cfg["v1_dir"])/"bars_features.parquet")
    s=fit_scale(bars,folds[0]["train_dates"])
    copy=bars.copy(); future=copy.trading_date.ge(folds[0]["forecast_dates"][0])
    from iclab.data import FEATURES
    copy.loc[future,FEATURES]=copy.loc[future,FEATURES]*999
    np.testing.assert_array_equal(s,fit_scale(copy,folds[0]["train_dates"]))
    d=json.loads((out/"state_definition.json").read_text()); tr=meta.trading_date.isin(folds[0]["train_dates"])
    assert d["rv_threshold_bps"]==np.median(meta.loc[tr,"current_rv"])
    a=np.load(out/"dataset.npz")
    for r in meta.iloc[::257].itertuples():
        ex=bars.iloc[r.entry_bar_id+5]
        assert ex.segment_id==r.segment_id
        assert np.isclose(a["y"][r.sample_id,0],(ex.open/bars.iloc[r.entry_bar_id].open-1)*10000)


def test_ensemble_does_not_mix_probes_or_raw_inputs():
    from diagnostics_v2.rolling import ensemble
    d={"stdp_s104__linear":np.ones((3,11)),"stdp_s211__linear":np.ones((3,11))*3,"stdp_s307__linear":np.ones((3,11))*5,"stdp_s104__rff":np.ones((3,11))*9,"raw_history__linear":np.ones((3,11))*7}
    e=ensemble(d)
    np.testing.assert_array_equal(e["stdp__linear"],np.ones((3,11))*3)
    np.testing.assert_array_equal(e["stdp__rff"],np.ones((3,11))*9)
    np.testing.assert_array_equal(e["raw_history__linear"],np.ones((3,11))*7)


def test_multiclass_probe_uses_shared_alpha_and_saved_model(tmp_path):
    from diagnostics_v2.rolling import read_models
    rng=np.random.default_rng(23);x=rng.normal(size=(80,7));v=rng.normal(size=(30,7))
    y=rng.normal(size=(80,11));yv=rng.normal(size=(30,11))
    m=fit_probe(x,v,y,yv,"linear",[1,10])
    assert len(set(m["alphas"][3:7]))==1 and len(set(m["alphas"][7:11]))==1
    np.savez_compressed(tmp_path/"raw_history__linear.npz",**m)
    loaded=read_models(tmp_path)["raw_history__linear"]
    np.testing.assert_array_equal(predict_probe(m,v),predict_probe(loaded,v))
