from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from iclab.data import FEATURES, dump_json, sha256
from sequence_v8.core import CompetitivePatterns
from event_stdp_v7.core import encode, price_events, samples, fit_stdp, shuffled_teacher, fit_ridge, lagged, score
from finance_v8.core import delayed_history, hidden_threshold, fit_autonomous

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = ROOT/'independent_study/results/v8_finance'


def main():
    cfg = json.loads((HERE/'config.json').read_text())
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError('Refuse to overwrite frozen experiment')
    OUT.mkdir(parents=True)
    sources = [ROOT/'independent_study/results/v1/bars_features.parquet',
               ROOT/'independent_study/results/v2/folds.json',
               ROOT/'independent_study/results/v2/backtests/rolling_all__raw_history__base_trades.csv',
               ROOT/'independent_study/configs/experiment.json',
               ROOT/'independent_study/sequence_v8/core.py',
               ROOT/'independent_study/event_stdp_v7/core.py',
               ROOT/'independent_study/risk_v4/core.py']
    sources += list(HERE.glob('*.py'))+[HERE/'config.json', HERE/'PROTOCOL.md']
    old_artifacts = {}
    for version in ['v7','v8_mechanism']:
        manifest = json.loads((ROOT/f'independent_study/results/{version}/manifest.json').read_text())
        for path, expected in manifest['artifacts'].items():
            assert sha256(ROOT/path)==expected,path
            old_artifacts[path]=expected
    hashes = {str(p.relative_to(ROOT)):sha256(p) for p in sources}
    dump_json(OUT/'source_lock.json', dict(config=cfg, sources=hashes, old_artifacts=old_artifacts))
    bars = pd.read_parquet(sources[0]); folds = json.loads(sources[1].read_text())
    features = bars[FEATURES].to_numpy(); seg = bars.segment_id.to_numpy(); close = bars.close.to_numpy()
    returns0, _, _ = price_events(close, seg, np.inf)
    finite = np.isfinite(features).all(axis=1)
    allframes, diagnostics, seedframes = [], [], []
    started = time.time()
    for fold in folds:
        fid = fold['fold']; fd = OUT/'folds'/f'fold_{fid:02d}'; fd.mkdir(parents=True)
        train = bars.trading_date.isin(fold['train_dates']).to_numpy()
        validation = bars.trading_date.isin(fold['validation_dates']).to_numpy()
        forecast = bars.trading_date.isin(fold['forecast_dates']).to_numpy()
        assert max(fold['train_dates']) < min(fold['validation_dates']) < min(fold['forecast_dates'])
        bounds = np.quantile(features[train & finite], cfg['feature_quantiles'],axis=0)
        event_threshold = float(np.quantile(np.abs(returns0[train & np.isfinite(returns0)]),cfg['event_quantile']))
        returns, raw, onset = price_events(close,seg,event_threshold,cfg['quiet_minutes'])
        spikes = encode(features,seg,bounds)
        x, past_valid = delayed_history(spikes,features,seg,cfg['delays'])
        eligible, target = samples(features,seg,returns,raw,onset,cfg['horizon'])
        valid = past_valid & lagged(np.ones(len(bars),bool),seg,-cfg['horizon'])
        ti, vi, fi = train & eligible, validation & eligible, forecast & eligible
        train_hidden = train & past_valid
        pieces = {k:[] for k in ['teacher','autonomous','random','shuffled','fixed13','equal_hidden']}
        for seed in cfg['seeds']:
            learned = CompetitivePatterns(x.shape[1],cfg,seed).fit(x[train_hidden],seed+1)
            random = CompetitivePatterns(x.shape[1],cfg,seed)
            theta = hidden_threshold(learned.weight,x[train_hidden],cfg['hidden_quantile'])
            rt = hidden_threshold(random.weight,x[train_hidden],cfg['hidden_quantile'])
            h = learned.response(x,theta)[0]; h[~past_valid]=0
            hr = random.response(x,rt)[0]; hr[~past_valid]=0
            hf = learned.response(x)[0]; hf[~past_valid]=0
            w, stats = fit_stdp(h,onset,seg,train,cfg['output'])
            wr, _ = fit_stdp(hr,onset,seg,train,cfg['output'])
            wf, _ = fit_stdp(hf,onset,seg,train,cfg['output'])
            fake = shuffled_teacher(onset,seg,train,seed)
            ws, _ = fit_stdp(h,fake,seg,train,cfg['output'])
            wa, auto = fit_autonomous(h,seg,train,cfg['output'])
            predictions = dict(teacher=h@w,autonomous=h@wa,random=hr@wr,
                               shuffled=h@ws,fixed13=hf@wf,equal_hidden=h@np.full(cfg['hidden'],.5))
            for name,p in predictions.items():
                pieces[name].append(p)
            np.savez_compressed(fd/f'seed_{seed}.npz', hidden=learned.weight, initial=random.weight,
                                theta=theta, random_theta=rt, teacher=w, random=wr,
                                fixed13=wf, shuffled=ws, autonomous=wa,
                                autonomous_theta=auto['final_theta'], **{f'teacher_{k}':v for k,v in stats.items()})
            norm=learned.weight/np.maximum(np.linalg.norm(learned.weight,axis=1,keepdims=True),1e-12)
            diagnostics.append(dict(fold=fid,seed=seed,train_input_spikes=float(x[train_hidden].sum(axis=1).mean()),
               train_rate=float(h[train_hidden].mean()), forecast_rate=float(h[forecast & past_valid].mean()),
               random_train_rate=float(hr[train_hidden].mean()),fixed_train_rate=float(hf[train_hidden].mean()),
               fixed_forecast_rate=float(hf[forecast & past_valid].mean()),active_forecast=int(np.count_nonzero(h[forecast & past_valid].sum(axis=0))),
               mean_cosine=float((norm@norm.T)[~np.eye(cfg['hidden'],dtype=bool)].mean()),
               teacher_w_mean=float(w.mean()),teacher_w_std=float(w.std()),theta_mean=float(theta.mean()),
               autonomous_theta=float(auto['final_theta'])))
            sf=bars.loc[fi,['bar_id','trading_date']].copy();sf['fold']=fid;sf['seed']=seed;sf['target']=target[fi]
            for name,p in predictions.items():sf[name]=p[fi]
            seedframes.append(sf)
        scores = {name:np.mean(values,axis=0) for name,values in pieces.items()}
        scores['delay_ridge'], ridge = fit_ridge(x,target,ti,vi,cfg)
        scores['current_rv'] = np.sqrt(sum(lagged(np.nan_to_num(returns)**2,seg,d) for d in range(5)))
        oldpath=ROOT/f'independent_study/results/v7/folds/fold_{fid:02d}/model.npz'
        old=np.load(oldpath)
        np.testing.assert_array_equal(bounds,old['bounds']);assert event_threshold==old['event_threshold']
        scores['v7'],_ = score(spikes,seg,old['weight'],cfg['output'])
        refs={name:np.sort(p[vi]) for name,p in scores.items()}
        rules={name:dict(q60=float(np.quantile(p[vi],.6)),q80=float(np.quantile(p[vi],.8))) for name,p in scores.items()}
        np.savez_compressed(fd/'preprocess.npz',bounds=bounds,event_threshold=event_threshold,
                            **{f'ridge_{k}':v for k,v in ridge.items()},**{f'ref_{k}':v for k,v in refs.items()})
        for split, mask in [('validation',validation & valid),('forecast',forecast & valid)]:
            frame=bars.loc[mask,['bar_id','segment_id','datetime','trading_date','contract_id']].copy()
            frame['fold']=fid;frame['covered']=eligible[mask];frame['target']=target[mask].astype(int)
            frame['event_now']=onset[mask];frame['event_threshold_bp']=event_threshold
            for name,p in scores.items():frame[name]=p[mask]
            frame.to_parquet(fd/f'{split}.parquet',index=False)
            if split=='forecast':allframes.append(frame)
        dump_json(fd/'lock.json',dict(fold=fold,rules=rules,train_hidden=int(train_hidden.sum()),
                  train_event_decisions=int(ti.sum()),validation_event_decisions=int(vi.sum()),
                  artifacts={p.name:sha256(p) for p in fd.iterdir() if p.is_file()}))
        print(f'fold {fid:02d}/22; event decisions={fi.sum()}, hidden firing={diagnostics[-1]["forecast_rate"]:.3f}; {time.time()-started:.1f}s',flush=True)
    forecasts=pd.concat(allframes,ignore_index=True)
    assert not forecasts.bar_id.duplicated().any()
    forecasts.to_parquet(OUT/'forecasts.parquet',index=False)
    pd.concat(seedframes,ignore_index=True).to_parquet(OUT/'seed_forecasts.parquet',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'neural_diagnostics.csv',index=False)
    for path,expected in hashes.items():assert sha256(ROOT/path)==expected,path
    dump_json(OUT/'training_status.json',dict(status='complete',folds=23,learned_hidden=69,random_hidden=69,
       output_stdp_models=345,ridge_models=23,forecast_minutes=len(forecasts),quiet_decisions=int(forecasts.covered.sum()),
       seconds=time.time()-started))


if __name__=='__main__':main()
