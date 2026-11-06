"""Analyze completed, locked v2 forecasts. Never trains or selects on evaluation data."""
from pathlib import Path
import json,re
import numpy as np
import pandas as pd
from iclab.analysis import evaluate,correlation,sufficient_daily,corr_sums
from iclab.backtest import simulate,trade_accounting
from iclab.data import sha256
from diagnostics_v2.common import config,save_json,classification,source_hashes


def load_predictions(out):
    metas=[];arrays={}
    for folder in sorted((out/"folds").glob("fold_*")):
        assert (folder/"complete.json").exists(),str(folder)
        done=json.loads((folder/"complete.json").read_text())
        assert sha256(folder/"predictions.npz")==done["predictions_sha256"]
        for arm,h in done["locks"].items(): assert sha256(folder/arm/"lock.json")==h
        metas.append(pd.read_parquet(folder/"samples.parquet"))
        a=np.load(folder/"predictions.npz")
        for key in a.files:
            if re.search(r"_s\d+__",key): continue
            arrays.setdefault(key,[]).append(a[key])
    meta=pd.concat(metas,ignore_index=True)
    assert meta.sample_id.is_unique and meta.datetime.is_monotonic_increasing
    return meta,{k:np.concatenate(v) for k,v in arrays.items()}


def metrics_for(key,p,y,meta):
    arm,rep,probe=key.split("__");common={"arm":arm,"representation":rep,"probe":probe}
    rows=[]
    for j,target in enumerate(["return_5","absolute_5","realized_volatility_5"]):
        rows.append({**common,"target":target,**evaluate(p[:,j],y[:,j],meta.trading_date.to_numpy(),risk=j!=0)})
    for target,start,label in [("current_state",3,meta.current_state.to_numpy()),("future_state",7,meta.future_state.to_numpy())]:
        m=classification(label,p[:,start:start+4]);rows.append({**common,"target":target,**{k:v for k,v in m.items() if k not in ["confusion","support"]}})
    return rows


def confusion_daily(scores,labels,dates,calendar):
    lookup={d:i for i,d in enumerate(calendar)}
    ix=np.array([lookup[d] for d in dates]);category=labels*4+scores.argmax(axis=1)
    result=np.zeros((len(calendar),16),dtype=np.int64)
    np.add.at(result,(ix,category),1)
    return result


def balanced_accuracy(confusions):
    cm=confusions.reshape(-1,4,4);support=cm.sum(axis=2);tp=np.diagonal(cm,axis1=1,axis2=2)
    recall=np.divide(tp,support,out=np.full(tp.shape,np.nan,dtype=float),where=support>0)
    return np.nanmean(recall,axis=1)


def state_details(pred,meta,out):
    """Descriptive decompositions of fixed scores; no new model or threshold fit."""
    confusion_rows=[];component_rows=[]
    for key,p in pred.items():
        if "__trade_" in key:continue
        arm,rep,probe=key.split("__")
        for target,start in [("current_state",3),("future_state",7)]:
            labels=meta[target].to_numpy();scores=p[:,start:start+4]
            cm=classification(labels,scores)["confusion"]
            for actual in range(4):
                for predicted in range(4):
                    confusion_rows.append({"model":key,"target":target,"actual":actual,"predicted":predicted,"n":int(cm[actual,predicted])})
            # The four mutually exclusive scores encode two binary properties.
            # Sum unnormalized class scores within each property; no refitting.
            for component,truth,binary in [
                ("high_volatility",labels//2,np.c_[scores[:,:2].sum(axis=1),scores[:,2:].sum(axis=1)]),
                ("high_efficiency",labels%2,np.c_[scores[:,[0,2]].sum(axis=1),scores[:,[1,3]].sum(axis=1)])]:
                m=classification(truth,binary,classes=2)
                component_rows.append({"arm":arm,"representation":rep,"probe":probe,"target":target,"component":component,**{k:v for k,v in m.items() if k not in ["confusion","support"]}})
    pd.DataFrame(confusion_rows).to_csv(out/"state_confusions.csv",index=False)
    pd.DataFrame(component_rows).to_csv(out/"state_components.csv",index=False)


def seed_details(out,meta,y):
    arrays={}
    for folder in sorted((out/"folds").glob("fold_*")):
        with np.load(folder/"predictions.npz") as a:
            for key in a.files:
                if re.search(r"_s\d+__",key):arrays.setdefault(key,[]).append(a[key])
    rows=[]
    for key,chunks in arrays.items():rows+=metrics_for(key,np.concatenate(chunks),y,meta)
    pd.DataFrame(rows).to_csv(out/"seed_metrics.csv",index=False)


def bootstrap(pred,y,meta,calendar,cfg):
    rng=np.random.default_rng(92231);n=len(calendar);b=cfg["bootstrap_block_days"]
    starts=rng.integers(0,max(n-b+1,1),(cfg["bootstrap_repetitions"],int(np.ceil(n/b))))
    indices=(starts[:,:,None]+np.arange(b)).reshape(len(starts),-1)[:,:n]
    intervals=[];distributions={};points={}
    for key,p in pred.items():
        if "__trade_" in key:continue
        dist={};point={}
        for j,target in enumerate(["return_5","absolute_5","realized_volatility_5"]):
            sufficient=sufficient_daily(p[:,j],y[:,j],meta.trading_date.to_numpy(),calendar)
            dist[target]=corr_sums(sufficient[indices].sum(axis=1));point[target]=correlation(p[:,j],y[:,j])
        for target,start,labels in [("current_state",3,meta.current_state.to_numpy()),("future_state",7,meta.future_state.to_numpy())]:
            cm=confusion_daily(p[:,start:start+4],labels,meta.trading_date.to_numpy(),calendar)
            dist[target]=balanced_accuracy(cm[indices].sum(axis=1));point[target]=float(balanced_accuracy(cm.sum(axis=0)[None,:])[0])
        distributions[key]=dist;points[key]=point
        for target,d in dist.items():
            finite=d[np.isfinite(d)]
            intervals.append({"model":key,"target":target,"statistic":"balanced_accuracy" if "state" in target else "ic","estimate":point[target],"ci_low":float(np.quantile(finite,.025)) if len(finite) else np.nan,"ci_high":float(np.quantile(finite,.975)) if len(finite) else np.nan})
    pairs=[]
    for arm in cfg["arms"]:
        for probe in cfg["probe_types"]:
            for left,right in [("stdp","random"),("stdp","shuffle"),("stdp","raw_history"),("raw_plus_stdp","raw_history"),("counts_minute","raw_history"),("counts_total","counts_minute")]:
                pairs.append((f"{arm}__{left}__{probe}",f"{arm}__{right}__{probe}"))
            for rep in ["stdp","raw_history"]:pairs.append((f"{arm}__{rep}__{probe}","baseline__persistence__rule"))
        for rep in ["raw_history","counts_minute","random","stdp","raw_plus_stdp"]:
            pairs.append((f"{arm}__{rep}__rff",f"{arm}__{rep}__linear"))
    for other in ["frozen","readout_only"]:
        for rep in ["raw_history","random","stdp","raw_plus_stdp"]:
            for probe in cfg["probe_types"]:pairs.append((f"rolling_all__{rep}__{probe}",f"{other}__{rep}__{probe}"))
    differences=[]
    for left,right in pairs:
        for target in distributions[left]:
            d=distributions[left][target]-distributions[right][target];finite=d[np.isfinite(d)]
            differences.append({"left":left,"right":right,"target":target,"difference":points[left][target]-points[right][target],"ci_low":float(np.quantile(finite,.025)) if len(finite) else np.nan,"ci_high":float(np.quantile(finite,.975)) if len(finite) else np.nan})
    return pd.DataFrame(intervals),pd.DataFrame(differences)


def backtests(cfg,v1,meta,pred,calendar,bars,out):
    records=[];dest=out/"backtests";dest.mkdir(exist_ok=True)
    for key,scores in pred.items():
        if "__trade_" not in key:continue
        arm,policy=key.split("__");policy=policy.removeprefix("trade_")
        for scenario,extra,ticks in [("base",0,1.),("two_ticks",0,2.),("extra_minute",1,1.)]:
            c={**v1["trading"],"slippage_ticks":ticks}
            summary,trades,equity,daily=simulate(bars,meta,scores,0.,c,calendar,extra_delay=extra)
            records.append({"arm":arm,"policy":policy,"scenario":scenario,**summary})
            prefix=arm+"__"+policy+"__"+scenario
            trades.to_csv(dest/(prefix+"_trades.csv"),index=False)
            daily.to_csv(dest/(prefix+"_daily.csv"),index=False)
            if scenario=="base":equity.to_parquet(dest/(prefix+"_equity.parquet"),index=False)
    pd.DataFrame(records).to_csv(out/"backtest_summary.csv",index=False)
    # Capital is kept by one continuous event simulation over all folds.
    selected=[]
    for folder in sorted((out/"folds").glob("fold_*")):
        for arm in cfg["arms"]:
            lock=json.loads((folder/arm/"lock.json").read_text())
            selected.append({"fold":lock["fold"]["fold"],"start":lock["fold"]["forecast_dates"][0],"end":lock["fold"]["forecast_dates"][-1],"arm":arm,"selected":lock["policy"]["selected"]})
    pd.DataFrame(selected).to_csv(out/"selected_policies.csv",index=False)


def verify(cfg,out):
    manifest=json.loads((out/"run_manifest.json").read_text())
    assert manifest["sources"]==source_hashes(),"Core source changed"
    prepared=json.loads((out/"prepared.json").read_text())
    checks={"v2_inputs_unchanged":all(sha256(p)==h for p,h in prepared["inputs"].items()),"core_sources_unchanged":True}
    equivalence=json.loads((out/"full_training_equivalence.json").read_text())
    checks["full_6000_window_3_epoch_reference_equivalence"]=equivalence["within_1e_6"]
    v1=Path(cfg["v1_dir"]);old=json.loads((v1/"artifact_manifest.json").read_text())
    checks["all_v1_artifacts_unchanged"]=all(sha256(v1/p)==h for p,h in old.items())
    checked=set();causal=True;models=True;frozen=True
    for folder in sorted((out/"folds").glob("fold_*")):
        for arm in cfg["arms"]:
            lock=json.loads((folder/arm/"lock.json").read_text());f=lock["fold"];eff=lock["effective_fit_dates"]
            causal &= max(eff["train_dates"])<min(eff["validation_dates"]) and max(eff["validation_dates"])<min(f["forecast_dates"])
            for p,h in lock["probe_hashes"].items():
                if p not in checked:models &=sha256(p)==h;checked.add(p)
            em=Path(lock["encoder_manifest"])
            models &= sha256(em)==lock["encoder_manifest_sha256"]
            if str(em) not in checked:
                enc=json.loads(em.read_text());models &=all(sha256(p)==h for p,h in enc["model_hashes"].items());frozen &=enc["frozen_inference_and_reload"];checked.add(str(em))
    checks.update(all_folds_fit_strictly_before_forecast=bool(causal),all_models_and_locks_intact=bool(models),all_encoder_freeze_reload_checks=bool(frozen))
    _,v1config=config()
    accounting=True;continuous=True
    for file in (out/"backtests").glob("*__base_trades.csv"):
        try:tr=pd.read_csv(file)
        except pd.errors.EmptyDataError:continue
        if len(tr):
            pnl=np.array([trade_accounting(r.entry_open,r.exit_open,r.side,v1config["trading"])["net_pnl"] for r in tr.itertuples()])
            accounting &= bool(np.allclose(pnl,tr.net_pnl,rtol=0,atol=1e-6))
            continuous &= bool(np.allclose(v1config["trading"]["capital"]+np.cumsum(pnl),tr.cash_after,rtol=0,atol=1e-6))
    checks["all_base_trade_costs_reconciled"]=accounting
    checks["capital_not_reset_between_folds"]=continuous
    if (out/"parallel_execution.json").exists():
        execution=json.loads((out/"parallel_execution.json").read_text())
        checks["parallel_helper_recorded_and_unchanged"]=sha256("independent_study/parallel_v2_tail.py")==execution["helper_sha256"]
    # Recreate saved forecasts from raw windows, checkpoint and probe, rather
    # than relying solely on file checksums or a repeated in-memory transform.
    import torch
    from diagnostics_v2.engine import FastSNN
    from diagnostics_v2.data import apply_scale
    from diagnostics_v2.probes import predict_probe
    from iclab.neural import encode
    torch.set_num_threads(1)
    raw=np.load(out/"dataset.npz")["raw"];reproduced=0
    for number in [0,10,22]:
        folder=out/"folds"/f"fold_{number:02d}"
        sm=pd.read_parquet(folder/"samples.parquet").head(32);saved=np.load(folder/"predictions.npz")
        for arm in cfg["arms"]:
            enc=folder/"rolling_all"/"encoder" if arm=="rolling_all" and number>0 else out/"initial_frozen"
            scale=np.load(enc/"scale.npy");x=apply_scale(raw[sm.sample_id.to_numpy()],scale)
            spikes=encode(x,sm.sample_id.to_numpy(),cfg["encoding"],104)
            model=FastSNN(24,v1config["network"],104).load(enc/"stdp_s104_snn.npz")
            representation=model.transform(spikes)
            pdirectory=out/"folds"/"fold_00"/"frozen"/"probes" if arm=="frozen" or number==0 else folder/arm/"probes"
            for kind in cfg["probe_types"]:
                probe=np.load(pdirectory/f"stdp_s104__{kind}.npz")
                prediction=predict_probe(probe,representation)
                np.testing.assert_allclose(prediction,saved[f"{arm}__stdp_s104__{kind}"][:len(sm)],rtol=1e-6,atol=1e-6)
                reproduced+=1
    checks["raw_to_saved_forecast_reproduction_18_cases"]=reproduced==18
    assert all(checks.values()),checks
    save_json(out/"verification.json",checks)


def main():
    cfg,v1=config();out=Path(cfg["output_dir"])
    assert (out/"rolling_complete.json").exists()
    meta,pred=load_predictions(out)
    y=np.load(out/"dataset.npz")["y"][meta.sample_id.to_numpy()]
    assert len(meta)==15900 and np.isfinite(y).all()
    calendar=pd.read_csv(Path(cfg["v1_dir"])/"contract_selection.csv").trading_date
    calendar=calendar[(calendar>=cfg["first_forecast"])&(calendar<=cfg["last_forecast"])].tolist()
    rows=[];monthly_rows=[];foldrows=[]
    for key,p in pred.items():
        if "__trade_" in key:continue
        rows+=metrics_for(key,p,y,meta)
        for month in sorted(meta.trading_date.str[:7].unique()):
            mask=meta.trading_date.str.startswith(month).to_numpy()
            monthly_rows += [{"month":month,**r} for r in metrics_for(key,p[mask],y[mask],meta[mask])]
        for fold in sorted(meta.fold.unique()):
            mask=meta.fold.eq(fold).to_numpy()
            foldrows += [{"fold":int(fold),**r} for r in metrics_for(key,p[mask],y[mask],meta[mask])]
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(monthly_rows).to_csv(out/"monthly_metrics.csv",index=False)
    pd.DataFrame(foldrows).to_csv(out/"fold_metrics.csv",index=False)
    state_details(pred,meta,out)
    seed_details(out,meta,y)
    intervals,diff=bootstrap(pred,y,meta,calendar,cfg)
    intervals.to_csv(out/"confidence_intervals.csv",index=False);diff.to_csv(out/"paired_comparisons.csv",index=False)
    bars=pd.read_parquet(Path(cfg["v1_dir"])/"bars_features.parquet")
    bars=bars[bars.trading_date.isin(calendar)]
    backtests(cfg,v1,meta,pred,calendar,bars,out)
    counts=[]
    for month,sub in meta.groupby(meta.trading_date.str[:7]):
        for target in ["current_state","future_state"]:
            for label in range(4):counts.append({"month":month,"target":target,"state":label,"n":int(sub[target].eq(label).sum()),"fraction":float(sub[target].eq(label).mean())})
    pd.DataFrame(counts).to_csv(out/"state_distribution.csv",index=False)
    # Rule to consider a future-state use: STDP must improve over persistence,
    # raw features and random SNN, not just recognize a state defined by inputs.
    gates=[]
    for probe in cfg["probe_types"]:
        left=f"rolling_all__stdp__{probe}";comparators=["baseline__persistence__rule",f"rolling_all__raw_history__{probe}",f"rolling_all__random__{probe}"]
        evidence=[]
        for right in comparators:
            row=diff[(diff.left==left)&(diff.right==right)&diff.target.eq("future_state")].iloc[0]
            evidence.append({"comparison":right,"difference":row.difference,"ci_low":row.ci_low,"ci_high":row.ci_high})
        gates.append({"probe":probe,"future_state_increment_supported":all(e["ci_low"]>0 for e in evidence),"evidence":evidence})
    save_json(out/"state_use_assessment.json",{"purpose":"Evidence assessment only; no test-set strategy selection","gates":gates,"next_action":"A state strategy needs separate design/evaluation if this gate passes; recognition alone is insufficient"})
    meta.to_parquet(out/"forecast_samples.parquet",index=False)
    np.savez_compressed(out/"ensemble_predictions.npz",**pred)
    verify(cfg,out)
    save_json(out/"analysis_sources.json",{"independent_study/analyze_v2.py":sha256("independent_study/analyze_v2.py")})
    save_json(out/"analysis_complete.json",{"forecast_rows":len(meta),"actual_days":int(meta.trading_date.nunique()),"package_days":len(calendar),"models_evaluated":len(rows)//5,"folds":int(meta.fold.nunique())})
    print(pd.DataFrame(rows).query("arm == 'rolling_all' and representation in ['raw_history','random','stdp']").to_string(index=False),flush=True)


if __name__=="__main__":main()
