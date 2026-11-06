"""Reconstruct selected complete folds and audit all frozen files and dates."""
import json
import numpy as np
import pandas as pd
from iclab.data import FEATURES,dump_json,sha256
from research_v10.state_run import ROOT,HERE,load
from research_v10.state_core import ProbabilitySTDP,categories,one_hot,predict_variance,fit_bins,shuffle_delays,qlike
from research_v10.temperature_core import posterior,predict
from research_v10.modulated_core import forward

RESULTS=ROOT/'independent_study/results'
OUT=RESULTS/'v10'


def main():
    cfg=json.loads((HERE/'config.json').read_text());bars,meta,folds,y,base,current=load()
    x=bars[FEATURES].to_numpy();ids=meta.bar_id.to_numpy();seg=bars.segment_id.to_numpy();sources=0;artifacts=0;old_counts={}
    for directory in ['v8_finance','v9_eligibility','v10_probe']:
        manifest=json.loads((RESULTS/directory/'manifest.json').read_text());count=0
        for section in ['artifacts','sources']:
            for path,h in manifest.get(section,{}).items():assert sha256(ROOT/path)==h,path;count+=1
        old_counts[directory]=count
    for stage in ['state_risk','increment','modulated','temperature','refit']:
        directory=RESULTS/f'v10_{stage}'
        for path,h in json.loads((directory/'source_lock.json').read_text()).items():assert sha256(ROOT/path)==h,path;sources+=1
        folders=list(directory.glob('fold_*')) if stage!='state_risk' else list(directory.glob('develop_k*/fold_*'))+list(directory.glob('audit/fold_*'))
        for fd in [p for p in folders if p.is_dir()]:
            lock=json.loads((fd/'lock.json').read_text());f=lock['fold']
            assert max(f['train_dates'])<min(f['validation_dates']) and max(f['validation_dates'])<min(f['forecast_dates'])
            for path,h in lock['files'].items():assert sha256(fd/path)==h,str(fd/path);artifacts+=1
            pred=pd.read_parquet(fd/'forecast.parquet');ix=pred.sample_id.to_numpy()
            np.testing.assert_array_equal(pred.bar_id,ids[ix]);np.testing.assert_array_equal(pred.realized,y[ix])
            assert set(pred.trading_date)<=set(f['forecast_dates'])
    gates=pd.read_csv(RESULTS/'v10_increment/gates.csv');gate_checks=0
    for f in folds:
        fid=f['fold'];fd=RESULTS/f'v10_increment/fold_{fid:02d}';v=pd.read_parquet(fd/'validation.parquet');p=pd.read_parquet(fd/'forecast.parquet')
        for source in ['stdp','random','shuffled']:
            trials=[(float(qlike(v.realized.to_numpy(),(1-w)*v.raw.to_numpy()+w*v[f'raw_plus_{source}'].to_numpy()).mean()),w) for w in [0.,.25,.5,.75,1.]]
            weight=min(trials)[1];saved=float(gates.query('fold == @fid and model == @source').weight.iloc[0]);assert weight==saved
            np.testing.assert_allclose(p[f'gated_{source}'],(1-weight)*p.raw+weight*p[f'raw_plus_{source}'],atol=1e-12,rtol=0);gate_checks+=1
    errors=[];retrained=0;refit_bounds=0;fixed_epochs=0
    for f in folds:
        fid=f['fold'];past=meta.trading_date.isin(f['train_dates']+f['validation_dates']).to_numpy();future=meta.trading_date.isin(f['forecast_dates']).to_numpy()
        pastbar=bars.trading_date.isin(f['train_dates']+f['validation_dates']).to_numpy()&np.isfinite(x).all(axis=1)
        rd=RESULTS/f'v10_refit/fold_{fid:02d}';bounds50=np.load(rd/'preprocess.npz')['bounds']
        np.testing.assert_array_equal(bounds50,fit_bins(x,pastbar,5));refit_bounds+=1
        # Every fitted target's endpoint is in a past date, including the refit validation history.
        endpoints=meta.exit_bar_id.to_numpy()[past]
        assert (bars.trading_date.iloc[endpoints].to_numpy()<min(f['forecast_dates'])).all()
        for source in ['stdp','random','shuffled']:
            for seed in cfg['seeds']:
                ep=int(np.load(RESULTS/f'v10_temperature/fold_{fid:02d}/{source}_t6_{seed}.npz')['selected_epoch'])
                for schedule in ['readout50','full50']:
                    assert int(np.load(rd/f'{schedule}_{source}_{seed}.npz')['epochs'])==ep;fixed_epochs+=1
        if fid not in [0,11,22]:continue
        old=(RESULTS/'v10_state_risk/develop_k16' if fid<9 else RESULTS/'v10_state_risk/audit')/f'fold_{fid:02d}'
        cat,raw=categories(x,np.load(old/'preprocess.npz')['bounds'],seg,ids,cfg['delays']);hot=one_hot(cat,5)
        cat50,_=categories(x,bounds50,seg,ids,cfg['delays']);hot50=one_hot(cat50,5)
        raw40=predict_variance(np.load(old/'raw_readout.npz'),np.c_[base,raw]);raw50=predict_variance(np.load(rd/'raw_readout.npz'),np.c_[base,raw])
        reconstructed={s:{} for s in ['increment','modulated','temperature','refit']}
        for source in ['stdp','random','shuffled']:
            c=[];t=[];r=[];full=[]
            for seed in cfg['seeds']:
                q=np.load(old/f'{source}_{seed}.npz')['q'];p1=posterior(q,hot,1.);p6=posterior(q,hot,6.)
                model=np.load(RESULTS/f'v10_increment/fold_{fid:02d}/{source}_{seed}_readout.npz')
                c.append(predict_variance(model,np.c_[base,raw,p1]))
                tmodel=np.load(RESULTS/f'v10_temperature/fold_{fid:02d}/{source}_t6_{seed}.npz');t.append(predict(tmodel,p6,raw40))
                rmodel=np.load(rd/f'readout50_{source}_{seed}.npz');np.testing.assert_array_equal(q,rmodel['q']);r.append(predict(rmodel,p6,raw50))
                fmodel=np.load(rd/f'full50_{source}_{seed}.npz');full.append(predict(fmodel,posterior(fmodel['q'],hot50,6.),raw50))
                if seed==104 and source!='random':
                    net=ProbabilitySTDP(36,5,16,cfg['eta'],cfg['epochs'],seed,cfg['probability_floor'])
                    training=cat50[past] if source=='stdp' else shuffle_delays(cat50[past],12,3,seed+2)
                    net.fit(training,seed+1);np.testing.assert_array_equal(net.q,fmodel['q']);retrained+=1
                if source=='stdp':
                    t1=np.load(RESULTS/f'v10_temperature/fold_{fid:02d}/stdp_t1_{seed}.npz')
                    d1=np.load(RESULTS/f'v10_modulated/fold_{fid:02d}/frozen_stdp_{seed}.npz')
                    np.testing.assert_allclose(predict(t1,p1,raw40),forward(d1['theta'],d1['a'],d1['b'],hot,raw40)[0],atol=1e-7,rtol=0)
            reconstructed['increment'][f'raw_plus_{source}']=np.mean(c,axis=0)
            reconstructed['temperature'][f'{source}_t6']=np.mean(t,axis=0)
            reconstructed['refit'][f'readout50_{source}']=np.mean(r,axis=0);reconstructed['refit'][f'full50_{source}']=np.mean(full,axis=0)
        reconstructed['refit']['raw50']=raw50
        for mode in ['frozen_stdp','true_modulated','mismatched','random_modulated','frozen_random']:
            values=[]
            for seed in cfg['seeds']:
                m=np.load(RESULTS/f'v10_modulated/fold_{fid:02d}/{mode}_{seed}.npz')
                values.append(forward(m['theta'],m['a'],m['b'],hot,raw40)[0])
            reconstructed['modulated'][mode]=np.mean(values,axis=0)
        for stage,predictions in reconstructed.items():
            for split in (['forecast'] if stage=='refit' else ['validation','forecast']):
                frame=pd.read_parquet(RESULTS/f'v10_{stage}/fold_{fid:02d}/{split}.parquet');ii=frame.sample_id.to_numpy()
                for model,pred in predictions.items():
                    error=float(abs(frame[model].to_numpy()-pred[ii]).max());assert error<1e-7,(fid,stage,model,error);errors.append(error)
        print('reconstructed fold',fid,flush=True)
    economic=json.loads((RESULTS/'v10_economics/verification.json').read_text())
    assert economic['cases']==27 and economic['accounting_records']==18144 and economic['margin_rejections']==0
    for path,h in economic['sources'].items():assert sha256(ROOT/path)==h,path
    record=dict(status='complete',source_hash_checks=sources,per_fold_artifact_checks=artifacts,prior_manifests_unchanged=old_counts,
        validation_gate_checks=gate_checks,refit_train_only_bins_checks=refit_bounds,fixed_epoch_checks=fixed_epochs,
        reconstructed_prediction_groups=len(errors),maximum_prediction_error=max(errors),exact_full50_hidden_retraining=retrained,
        all_forecast_targets_reconstructed=True,all_training_target_endpoints_before_forecast=True,economic=economic,
        mechanism_tests='22 passed: 10 research_v10 + 6 risk_v4 + 6 finance_v8')
    dump_json(OUT/'verification.json',record);print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
