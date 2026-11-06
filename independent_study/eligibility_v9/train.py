from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from iclab.data import FEATURES, dump_json, sha256
from event_stdp_v7.core import encode, price_events, samples, fit_stdp, lagged
from finance_v8.core import delayed_history, hidden_threshold
from eligibility_v9.core import EventModulatedPatterns, matured_feedback, mismatch_feedback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
OUT=ROOT/'independent_study/results/v9_eligibility'
OLD=ROOT/'independent_study/results/v8_finance'
ARMS=['unsupervised','true_feedback','mismatched']


def main():
    cfg=json.loads((HERE/'config.json').read_text())
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError('Refuse to overwrite experiment')
    OUT.mkdir(parents=True)
    sources=[ROOT/'independent_study/results/v1/bars_features.parquet',
             ROOT/'independent_study/results/v2/folds.json',
             ROOT/'independent_study/results/v2/backtests/rolling_all__raw_history__base_trades.csv',
             ROOT/'independent_study/configs/experiment.json']
    for module in ['sequence_v8/core.py','event_stdp_v7/core.py','risk_v4/core.py',
                   'finance_v8/core.py','finance_v8/controls.py']:
        sources.append(ROOT/'independent_study'/module)
    sources+=list(HERE.glob('*.py'))+[HERE/'config.json',HERE/'PROTOCOL.md']
    old_artifacts=json.loads((OLD/'manifest.json').read_text())['artifacts']
    old_artifacts={**old_artifacts,str((OLD/'manifest.json').relative_to(ROOT)):sha256(OLD/'manifest.json')}
    for path,expected in old_artifacts.items():assert sha256(ROOT/path)==expected,path
    hashes={str(p.relative_to(ROOT)):sha256(p) for p in sources}
    dump_json(OUT/'source_lock.json',dict(config=cfg,sources=hashes,old_artifacts=old_artifacts))
    bars=pd.read_parquet(sources[0]);folds=json.loads(sources[1].read_text())
    features=bars[FEATURES].to_numpy();seg=bars.segment_id.to_numpy();close=bars.close.to_numpy()
    ret0,_,_=price_events(close,seg,np.inf);finite=np.isfinite(features).all(axis=1)
    frames=[];seedframes=[];diags=[];units=[];probe=[];feedback_audit=[];start=time.time()
    for fold in folds:
        fid=fold['fold'];fd=OUT/'folds'/f'fold_{fid:02d}';fd.mkdir(parents=True)
        oldfd=OLD/'folds'/fd.name
        train=bars.trading_date.isin(fold['train_dates']).to_numpy()
        val=bars.trading_date.isin(fold['validation_dates']).to_numpy()
        forecast=bars.trading_date.isin(fold['forecast_dates']).to_numpy()
        bounds=np.quantile(features[train&finite],cfg['feature_quantiles'],axis=0)
        threshold=float(np.quantile(abs(ret0[train&np.isfinite(ret0)]),cfg['event_quantile']))
        ret,raw,onset=price_events(close,seg,threshold,cfg['quiet_minutes'])
        s=encode(features,seg,bounds);x,past_valid=delayed_history(s,features,seg,cfg['delays'])
        eligible,y=samples(features,seg,ret,raw,onset,cfg['horizon'])
        mature,feedback,p=matured_feedback(onset,seg,train,eligible,cfg['horizon'])
        np.testing.assert_array_equal(feedback[mature],y[mature])
        valid=past_valid&lagged(np.ones(len(bars),bool),seg,-cfg['horizon'])
        th=train&past_valid;ti=train&eligible;vi=val&eligible;fi=forecast&eligible
        pieces={name:[] for name in ARMS}
        for seed in cfg['seeds']:
            wrong=mismatch_feedback(feedback,seg,mature,seed)
            assert np.nansum(wrong)==np.nansum(feedback)
            feedback_audit.append(dict(fold=fid,seed=seed,feedback_n=int(mature.sum()),p_train=p,
                positive_n=int(feedback[mature].sum()),mismatch_fraction=float(np.mean(wrong[mature]!=feedback[mature]))))
            old=np.load(oldfd/f'seed_{seed}.npz');baseline_hidden=None
            sf=bars.loc[fi,['bar_id','trading_date']].copy();sf['fold']=fid;sf['seed']=seed;sf['target']=y[fi]
            for arm in ARMS:
                f=None if arm=='unsupervised' else (feedback if arm=='true_feedback' else wrong)[th]
                model=EventModulatedPatterns(x.shape[1],cfg,seed).fit(x[th],seed+1,f,p)
                theta=hidden_threshold(model.weight,x[th],cfg['hidden_quantile'])
                h=model.response(x,theta)[0];h[~past_valid]=0
                w,stats=fit_stdp(h,onset,seg,train,cfg['output'])
                prediction=h@w;pieces[arm].append(prediction);sf[arm]=prediction[fi]
                if arm=='unsupervised':
                    for a,b in [(model.weight,old['hidden']),(theta,old['theta']),(w,old['teacher'])]:
                        np.testing.assert_array_equal(a,b)
                    baseline_hidden=model.weight.copy()
                np.savez_compressed(fd/f'{arm}_{seed}.npz',hidden=model.weight,initial=model.initial_weight,
                    theta=theta,output=w,**{f'output_{k}':v for k,v in stats.items()})
                norm=model.weight/np.maximum(np.linalg.norm(model.weight,axis=1,keepdims=True),1e-12)
                cosine=(norm@norm.T)[~np.eye(cfg['hidden'],dtype=bool)].mean()
                diags.append(dict(fold=fid,seed=seed,model=arm,train_rate=float(h[th].mean()),
                    forecast_rate=float(h[forecast&past_valid].mean()),active_forecast=int((h[forecast].sum(axis=0)>0).sum()),
                    cosine=float(cosine),hidden_l1_change=float(abs(model.weight-baseline_hidden).mean()),
                    output_mean=float(w.mean()),output_std=float(w.std()),theta_mean=float(theta.mean()),**model.audit))
                nt=h[ti].sum(axis=0);nf=h[fi].sum(axis=0)
                pt=np.divide(h[ti].T@y[ti],nt,out=np.full(cfg['hidden'],np.nan),where=nt>0)
                pf=np.divide(h[fi].T@y[fi],nf,out=np.full(cfg['hidden'],np.nan),where=nf>0)
                best=int(np.argmax(np.where(nt>=20,pt-y[ti].mean(),-np.inf)))
                for j in range(cfg['hidden']):
                    units.append(dict(fold=fid,seed=seed,model=arm,unit=j,train_active=nt[j],
                        train_event_probability=pt[j],train_base=float(y[ti].mean()),forecast_active=nf[j],
                        forecast_event_probability=pf[j],forecast_base=float(y[fi].mean()),selected=(j==best)))
                pg=bars.loc[fi,['bar_id','trading_date']].copy();pg['fold']=fid;pg['seed']=seed;pg['model']=arm
                pg['target']=y[fi];pg['response']=h[fi,best];pg['selected_unit']=best;probe.append(pg)
            seedframes.append(sf)
        scores={k:np.mean(v,axis=0) for k,v in pieces.items()}
        refs={k:np.sort(v[vi]) for k,v in scores.items()}
        rules={k:dict(q60=float(np.quantile(v[vi],.6)),q80=float(np.quantile(v[vi],.8))) for k,v in scores.items()}
        oldp=np.load(oldfd/'preprocess.npz');oldlock=json.loads((oldfd/'lock.json').read_text())
        for k in ['current_rv','delay_ridge']:
            refs[k]=oldp[f'ref_{k}'];rules[k]=oldlock['rules'][k]
        np.savez_compressed(fd/'preprocess.npz',bounds=bounds,event_threshold=threshold,p_train=p,
            feedback_bar_ids=np.flatnonzero(mature),feedback_target=feedback[mature],
            **{f'ref_{k}':v for k,v in refs.items()})
        for split,mask in [('validation',val&valid),('forecast',forecast&valid)]:
            oldframe=pd.read_parquet(oldfd/f'{split}.parquet')
            frame=oldframe[['bar_id','segment_id','datetime','trading_date','contract_id','fold','covered',
                            'target','event_now','event_threshold_bp']].copy()
            np.testing.assert_array_equal(frame.bar_id,bars.bar_id[mask])
            np.testing.assert_array_equal(frame.covered,eligible[mask]);np.testing.assert_array_equal(frame.target,y[mask])
            for k,v in scores.items():frame[k]=v[mask]
            np.testing.assert_array_equal(frame.unsupervised,oldframe.teacher)
            for k in ['current_rv','delay_ridge']:frame[k]=oldframe[k]
            frame.to_parquet(fd/f'{split}.parquet',index=False)
            if split=='forecast':frames.append(frame)
        dump_json(fd/'lock.json',dict(fold=fold,rules=rules,p_train=p,matured_feedback=int(mature.sum()),
            artifacts={f.name:sha256(f) for f in fd.iterdir() if f.is_file()}))
        print(f'fold {fid:02d}/22; p={p:.4f}; exact v8 baseline; {time.time()-start:.1f}s',flush=True)
    result=pd.concat(frames,ignore_index=True);assert not result.bar_id.duplicated().any()
    result.to_parquet(OUT/'forecasts.parquet',index=False)
    pd.concat(seedframes,ignore_index=True).to_parquet(OUT/'seed_forecasts.parquet',index=False)
    pd.concat(probe,ignore_index=True).to_parquet(OUT/'hidden_selected_probe.parquet',index=False)
    pd.DataFrame(diags).to_csv(OUT/'neural_diagnostics.csv',index=False)
    pd.DataFrame(units).to_csv(OUT/'unit_associations.csv',index=False)
    pd.DataFrame(feedback_audit).to_csv(OUT/'feedback_audit.csv',index=False)
    for path,expected in hashes.items():assert sha256(ROOT/path)==expected,path
    dump_json(OUT/'training_status.json',dict(status='complete',folds=len(folds),hidden_models=207,
        output_models=207,baseline_exact_models=69,forecast_minutes=len(result),quiet_decisions=int(result.covered.sum()),
        seconds=time.time()-start))


if __name__=='__main__':main()
