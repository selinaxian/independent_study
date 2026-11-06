import numpy as np
import pandas as pd
import torch
from iclab.data import feature_segment, select_contracts
from iclab.neural import CompetitiveSNN, encode, stdp_delta
from iclab.analysis import fit_readout, predict_readout
from iclab.backtest import simulate, trade_accounting


def config():
    import json
    return json.load(open("independent_study/configs/experiment.json"))


def synthetic_day(day="2026-02-03", contract="IC2606"):
    dt = pd.date_range(day+" 09:31", periods=120, freq="min", tz="Asia/Shanghai").append(pd.date_range(day+" 13:01", periods=120, freq="min", tz="Asia/Shanghai"))
    c = 7000+np.sin(np.arange(240)*.3)*5+np.arange(240)*.1
    return pd.DataFrame({"datetime": dt, "trading_date": day, "session": ["AM"]*120+["PM"]*120, "contract_id": contract, "open": c-.1, "close": c, "high": c+1, "low": c-1, "volume": 100, "open_interest": 10000, "bar_id": np.arange(240), "segment_id": [1]*120+[2]*120})


def test_features_causal_and_prior_levels():
    g = synthetic_day().iloc[:120].copy()
    f = feature_segment(g)
    other = g.copy()
    other.loc[61:, ["open","high","low","close","volume","open_interest"]] *= 3
    np.testing.assert_allclose(f.iloc[:61], feature_segment(other).iloc[:61], equal_nan=True)
    # Current high cannot change the previous-20 resistance distance.
    other = g.copy(); other.loc[60, "high"] += 100
    assert f.loc[60,"resistance_distance"] == feature_segment(other).loc[60,"resistance_distance"]


def test_contract_selection_previous_day_only_and_no_fallback():
    a = synthetic_day("2026-02-03")
    b = synthetic_day("2026-02-03", "IC2609"); b.open_interest = 9000
    today = synthetic_day("2026-02-04"); today.open_interest = 1
    other = synthetic_day("2026-02-04", "IC2609"); other.open_interest = 999999
    _, d = select_contracts(pd.concat([a,b,today,other]), ["2026-02-03","2026-02-04"], config()["selection"])
    assert d.iloc[-1].contract_id == "IC2606"
    _, d = select_contracts(pd.concat([a,b,other]), ["2026-02-03","2026-02-04"], config()["selection"])
    assert d.iloc[-1].status == "selected_contract_missing_or_incomplete"


def test_encodings_are_bounded_reproducible_and_batch_independent():
    x = np.random.default_rng(10).uniform(-1,1,(10,12,12)).astype(np.float32)
    ids = np.arange(10)
    a = encode(x, ids, "rate", 104)
    assert a.dtype == bool and a.shape == (10,96,24)
    np.testing.assert_array_equal(a[4:], encode(x[4:],ids[4:],"rate",104))
    d = encode(x, ids, "density",104)
    mass = np.concatenate([np.maximum(x,0),np.maximum(-x,0)],axis=-1).sum(axis=1)*8*.3
    assert np.max(np.abs(d.sum(axis=1)-mass)) < 1.00001
    shuffled = encode(x,ids,"rate",104,shuffled=True)
    np.testing.assert_array_equal(a[:,-8:],shuffled[:,-8:])
    np.testing.assert_array_equal(a.sum(axis=1),shuffled.sum(axis=1))


def test_stdp_potentiation_depression_and_no_zero_lag():
    w = torch.full((1,1),.5); one=torch.ones(1,1); zero=torch.zeros(1,1)
    assert stdp_delta(w,one,zero,zero,one,.01,.01).item()>0
    assert stdp_delta(w,zero,one,one,zero,.01,.01).item()<0
    assert stdp_delta(w,zero,zero,one,one,.01,.01).item()==0
    doubled = stdp_delta(w,one.repeat(2,1),zero.repeat(2,1),zero.repeat(2,1),one.repeat(2,1),.01,.01)
    torch.testing.assert_close(doubled,stdp_delta(w,one,zero,zero,one,.01,.01))


def test_random_weights_frozen_and_inference_reset_reload(tmp_path):
    c = config()["network"].copy(); c.update(epochs=1,batch_size=4)
    model = CompetitiveSNN(24,c,104)
    spikes = np.random.default_rng(1).random((8,96,24))<.1
    model.fit(spikes,plastic=False)
    torch.testing.assert_close(model.weight,model.initial_weight,rtol=0,atol=0)
    w,t = model.weight.clone(),model.theta.clone()
    a=model.transform(spikes); b=model.transform(spikes)
    np.testing.assert_array_equal(a,b)
    torch.testing.assert_close(w,model.weight,rtol=0,atol=0)
    torch.testing.assert_close(t,model.theta,rtol=0,atol=0)
    model.save(tmp_path/"model.npz")
    np.testing.assert_array_equal(a,CompetitiveSNN(24,c,104).load(tmp_path/"model.npz").transform(spikes))
    assert np.max(a[:,:64].sum(axis=1))<=1.000001


def test_readout_scaling_train_only():
    rng=np.random.default_rng(1); tr=rng.normal(size=(50,3)); va=rng.normal(size=(20,3))+20
    yt=rng.normal(size=(50,5)); yv=rng.normal(size=(20,5))
    m=fit_readout(tr,va,yt,yv,[1,10])
    np.testing.assert_allclose(m["mean"],tr.mean(axis=0))
    assert np.all(predict_readout(m,va)[:,-1]>=0)


def test_costs_long_and_short():
    c=config()["trading"]
    for side in [1,-1]:
        a=trade_accounting(7000,7001,side,c)
        assert np.isclose(a["gross_pnl"],200*side)
        assert np.isclose(a["slippage_cost"],80)
        expected=(7000+side*.2)*200*.000033+1+(7001-side*.2)*200*.00024+1
        assert np.isclose(a["net_pnl"],200*side-80-expected)


def test_delayed_execution_nonoverlap_margin_and_flat():
    bars=synthetic_day().iloc[:120]
    ids=np.arange(35,90)
    meta=pd.DataFrame({"sample_id":ids,"bar_id":ids,"entry_bar_id":ids+2})
    c=config()["trading"]
    s,tr,eq,_=simulate(bars,meta,np.ones(len(ids)),0,c,["2026-02-03"])
    assert len(tr)>0 and (tr.entry_bar_id==tr.signal_bar_id+2).all()
    assert (tr.entry_time>tr.signal_time).all()
    assert (tr.entry_bar_id.to_numpy()[1:]>tr.exit_bar_id_actual.to_numpy()[:-1]).all()
    assert np.isclose(tr.net_pnl.sum(),s["net_pnl"])
    poor=c.copy(); poor["capital"]=1000
    s,*_=simulate(bars,meta,np.ones(len(ids)),0,poor,["2026-02-03"])
    assert s["trades"]==0 and s["margin"]>0
    s,*_=simulate(bars,meta,np.ones(len(ids)),0,c,["2026-02-03"],enabled=False)
    assert s["net_pnl"]==0 and s["max_drawdown_pct"]==0


def test_actual_dataset_boundaries_and_label_alignment():
    from pathlib import Path
    p=Path(config()["output_dir"])
    if not (p/"dataset.npz").exists(): return
    a=np.load(p/"dataset.npz"); meta=pd.read_parquet(p/"samples.parquet"); bars=pd.read_parquet(p/"bars_features.parquet").set_index("bar_id")
    assert np.isfinite(a["X"]).all() and np.isfinite(a["y"]).all()
    for row in meta.iloc[::83].itertuples():
        window=bars.loc[row.bar_id-11:row.bar_id]
        assert window.segment_id.nunique()==1 and window.contract_id.nunique()==1
        assert window.datetime.diff().dropna().eq(pd.Timedelta(minutes=1)).all()
        ent,ex=bars.loc[row.entry_bar_id],bars.loc[row.exit_bar_id]
        assert ent.segment_id==ex.segment_id==row.segment_id
        assert ent.datetime-pd.Timedelta(minutes=1)>row.datetime
        assert np.isclose(a["y"][row.sample_id,2],(ex.open/ent.open-1)*10000)
