"""Run the predeclared 23-fold teacher-STDP event-warning prototype."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from iclab.data import FEATURES, dump_json, sha256
from event_stdp_v7.core import (lagged, encode, price_events, samples, fit_stdp,
                               score, shuffled_teacher, classification, fit_ridge)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = ROOT/'independent_study/results/v7'


def event_rows(bars, onset, eligible, forecast, alarms, fold, horizon):
    rows = []
    segments = bars.segment_id.to_numpy()
    for t in np.flatnonzero(onset & forecast):
        candidates = np.arange(max(0, t-horizon), t)
        candidates = candidates[(segments[candidates] == segments[t]) & eligible[candidates] & forecast[candidates]]
        for model, alert in alarms.items():
            hits = candidates[alert[candidates]]
            rows.append(dict(fold=fold, model=model, event_bar_id=int(t),
                             trading_date=bars.trading_date.iloc[t],
                             event_datetime=bars.datetime.iloc[t],
                             evaluable=bool(len(candidates)), detected=bool(len(hits)),
                             earliest_lead_minutes=int(t-hits.min()) if len(hits) else np.nan,
                             latest_lead_minutes=int(t-hits.max()) if len(hits) else np.nan))
    return rows


def main():
    started = time.time()
    cfg = json.loads((HERE/'config.json').read_text())
    source = ROOT/'independent_study/results/v1/bars_features.parquet'
    foldpath = ROOT/'independent_study/results/v2/folds.json'
    folds = json.loads(foldpath.read_text())
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError('Refuse to overwrite v7 artifacts; use a new experiment directory')
    OUT.mkdir(parents=True, exist_ok=True)
    source_paths = [source, foldpath, HERE/'config.json', HERE/'PROTOCOL.md', HERE/'core.py', HERE/'run.py']
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    dump_json(OUT/'run_lock.json', dict(config=cfg, hashes=hashes, status='locked_before_fit'))
    bars = pd.read_parquet(source)
    assert np.array_equal(bars.bar_id, np.arange(len(bars)))
    assert bars.datetime.is_monotonic_increasing
    segments = bars.segment_id.to_numpy()
    x = bars[FEATURES].to_numpy(dtype=float)
    close = bars.close.to_numpy(dtype=float)
    base_returns, _, _ = price_events(close, segments, np.inf)
    channels = [f'{f}:{side}' for f in FEATURES for side in ['low', 'high']]
    all_predictions, weights_rows, lead_rows, fold_summaries = [], [], [], []
    for fold in folds:
        fid = fold['fold']
        train = bars.trading_date.isin(fold['train_dates']).to_numpy()
        val = bars.trading_date.isin(fold['validation_dates']).to_numpy()
        future = bars.trading_date.isin(fold['forecast_dates']).to_numpy()
        assert max(fold['train_dates']) < min(fold['validation_dates'])
        assert max(fold['validation_dates']) < min(fold['forecast_dates'])
        finite = np.isfinite(x).all(axis=1)
        bounds = np.quantile(x[train & finite], cfg['feature_quantiles'], axis=0)
        threshold = float(np.quantile(np.abs(base_returns[train & np.isfinite(base_returns)]), cfg['event_quantile']))
        returns, raw_events, onset = price_events(close, segments, threshold, cfg['quiet_minutes'])
        spikes = encode(x, segments, bounds)
        eligible, y = samples(x, segments, returns, raw_events, onset, cfg['horizon'])
        ti, vi, fi = train & eligible, val & eligible, future & eligible
        assert ti.sum() and vi.sum() and fi.sum()
        w, stats = fit_stdp(spikes, onset, segments, train, cfg)
        stdp_score, currents = score(spikes, segments, w, cfg)
        shuffled_weights, shuffled_scores = [], []
        model_stats = [('teacher_stdp', w, stats)]
        for seed in cfg['shuffle_seeds']:
            fake = shuffled_teacher(onset, segments, train, seed)
            sw, ss = fit_stdp(spikes, fake, segments, train, cfg)
            shuffled_weights.append(sw)
            shuffled_scores.append(currents @ sw)
            model_stats.append((f'shuffled_{seed}', sw, ss))
        rv = np.sqrt(sum(lagged(np.nan_to_num(returns)**2, segments, d) for d in range(5)))
        ridge_score, ridge = fit_ridge(currents, y, ti, vi, cfg)
        scores = dict(teacher_stdp=stdp_score,
                      untrained_equal=currents @ np.full(24, cfg['initial_weight']),
                      shuffled_teacher=np.mean(shuffled_scores, axis=0),
                      current_rv=rv, same_input_ridge=ridge_score)
        alarms, cutoffs, validation_sorted = {}, {}, {}
        for model, s in scores.items():
            cutoff = float(np.quantile(s[vi], cfg['alarm_quantile']))
            ref = np.sort(s[vi])
            # Past-only common scale across rolling models, not probability.
            percentile = np.searchsorted(ref, s, side='right')/len(ref)
            alarm = s > cutoff
            alarms[model], cutoffs[model], validation_sorted[model] = alarm, cutoff, ref
            frame = bars.loc[fi, ['bar_id','segment_id','datetime','trading_date','contract_id']].copy()
            frame['fold'], frame['model'] = fid, model
            frame['target'] = y[fi].astype(int)
            frame['score_raw'], frame['score'], frame['alarm'] = s[fi], percentile[fi], alarm[fi]
            frame['event_threshold_bp'], frame['alarm_threshold'] = threshold, cutoff
            all_predictions.append(frame)
            fold_summaries.append(dict(fold=fid, model=model, train_decisions=int(ti.sum()),
                                       validation_decisions=int(vi.sum()), event_threshold_bp=threshold,
                                       train_events=int(onset[train].sum()),
                                       **classification(y[fi], percentile[fi], alarm[fi])))
        for model, ww, st in model_stats:
            for i, channel in enumerate(channels):
                weights_rows.append(dict(fold=fid, model=model, channel=channel,
                                         weight=float(ww[i]), input_spikes=int(spikes[train, i].sum()),
                                         **{key: float(value[i]) for key, value in st.items()}))
        lead_rows.extend(event_rows(bars, onset, eligible, future, alarms, fid, cfg['horizon']))
        folder = OUT/'folds'/f'fold_{fid:02d}'
        folder.mkdir(parents=True)
        np.savez_compressed(folder/'model.npz', weight=w, shuffled_weights=np.array(shuffled_weights),
                            bounds=bounds, event_threshold=threshold,
                            **{f'ridge_{k}': v for k, v in ridge.items()},
                            **{f'validation_sorted_{k}': v for k, v in validation_sorted.items()})
        dump_json(folder/'lock.json', dict(fold=fold, cutoffs=cutoffs, config=cfg,
                                          model_sha256=sha256(folder/'model.npz')))
        print(f'fold {fid:02d}: train={ti.sum()}, forecast={fi.sum()}, future-event fraction={y[fi].mean():.3f}', flush=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    assert not predictions.duplicated(['model','bar_id']).any()
    predictions.to_parquet(OUT/'predictions.parquet', index=False)
    pd.DataFrame(weights_rows).to_csv(OUT/'weights.csv', index=False)
    pd.DataFrame(lead_rows).to_csv(OUT/'event_leads.csv', index=False)
    pd.DataFrame(fold_summaries).to_csv(OUT/'fold_metrics.csv', index=False)
    for path, expected in hashes.items():
        assert sha256(ROOT/path) == expected, path
    dump_json(OUT/'run_status.json', dict(status='training_complete', folds=len(folds),
              minutes=len(bars), forecast_decisions=int((predictions.model == 'teacher_stdp').sum()),
              trained_weight_sets=len(folds)*(1+len(cfg['shuffle_seeds'])),
              runtime_seconds=time.time()-started))


if __name__ == '__main__':
    main()
