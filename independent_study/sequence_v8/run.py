from pathlib import Path
import json
import numpy as np
import pandas as pd
from iclab.data import dump_json, sha256
from event_stdp_v7.core import fit_stdp, classification
from sequence_v8.core import (CompetitivePatterns, delayed_inputs, generate,
                              output_training_arrays, KINDS)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = ROOT/'independent_study/results/v8_mechanism'


def noisy(data, seed):
    result = {k: v.copy() for k, v in data.items()}
    rng = np.random.default_rng(seed)
    other = np.setdiff1d(np.arange(result['spikes'].shape[-1]), result['identities'])
    for i, anchor in enumerate(result['anchors']):
        result['spikes'][i, anchor-int(rng.integers(0,5)), int(rng.choice(other))] = 1
    assert np.all(result['spikes'].sum(axis=(1,2)) == 3)
    return result


def permuted_events(data, seed):
    out = {k: v.copy() for k, v in data.items()}
    out['events'][:] = False
    order = np.random.default_rng(seed).permutation(len(data['anchors']))
    for i, source in enumerate(order):
        for t in np.flatnonzero(data['events'][source]):
            dt = t-data['anchors'][source]
            out['events'][i, data['anchors'][i]+dt] = True
    return out


def main():
    cfg = json.loads((HERE/'config.json').read_text())
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError('Refuse to overwrite existing mechanism artifacts')
    OUT.mkdir(parents=True)
    paths = list(HERE.glob('*.py')) + [HERE/'config.json', HERE/'DESIGN.md',
                                       ROOT/'independent_study/event_stdp_v7/core.py',
                                       ROOT/'independent_study/results/v7/manifest.json']
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    dump_json(OUT/'lock.json', dict(config=cfg, hashes=hashes,
              scope='Synthetic mechanism only; no financial fit or backtest'))
    metric_rows, pattern_rows, diagnostics, prediction_rows = [], [], [], []
    for seed, identities in zip(cfg['seeds'], cfg['identities']):
        train = generate(cfg['train_per_kind'], seed+10000, identities, cfg['channels'])
        val = generate(cfg['validation_per_kind'], seed+20000, identities, cfg['channels'])
        test = generate(cfg['test_per_kind'], seed+30000, identities, cfg['channels'])
        noise = noisy(test, seed+40000)
        datasets = dict(train=train, validation=val, clean=test, noise=noise)
        delayed = {k: delayed_inputs(d['spikes'], d['anchors'], cfg['delays']) for k,d in datasets.items()}
        counts = {k: x.reshape(len(x),cfg['channels'],-1).sum(axis=2) for k,x in delayed.items()}
        hidden = CompetitivePatterns(delayed['train'].shape[1], cfg, seed).fit(delayed['train'], seed+1)
        count_hidden = CompetitivePatterns(cfg['channels'], cfg, seed).fit(counts['train'], seed+1)
        random = CompetitivePatterns(delayed['train'].shape[1], cfg, seed)
        target_activity = hidden.response(delayed['train'])[0].mean()
        random_drive = random.response(delayed['train'])[1]
        matched_threshold = float(np.quantile(random_drive, 1-target_activity))
        representations = {
            'learned_delays': {k: hidden.response(x)[0] for k,x in delayed.items()},
            'random_fixed': {k: random.response(x)[0] for k,x in delayed.items()},
            'random_activity_matched': {k: random.response(x, matched_threshold)[0] for k,x in delayed.items()},
            'learned_counts_hidden': {k: count_hidden.response(x)[0] for k,x in counts.items()},
            'direct_counts': counts,
        }
        representations['learned_delays_shuffled_teacher'] = representations['learned_delays']
        folder = OUT/f'seed_{seed}'
        folder.mkdir()
        np.savez_compressed(folder/'hidden.npz', learned=hidden.weight, counts=count_hidden.weight,
                            initial=random.weight, usage=hidden.usage,
                            random_matched_threshold=matched_threshold, identities=identities)
        for name, rep in representations.items():
            teacher_data = permuted_events(train, seed+50000) if name.endswith('shuffled_teacher') else train
            pre, post, segments = output_training_arrays(rep['train'], teacher_data)
            output, stats = fit_stdp(pre, post, segments, np.ones(len(post),bool), cfg['output'])
            np.savez_compressed(folder/f'{name}_output.npz', weight=output, **stats)
            val_score = rep['validation'] @ output
            cutoff = float(np.quantile(val_score, cfg['alarm_quantile']))
            for test_name in ['clean','noise']:
                d = datasets[test_name]
                score = rep[test_name] @ output
                for subset, mask in [('all',np.ones(len(score),bool)), ('same_AB_counts',d['kinds'] < 5)]:
                    metric_rows.append(dict(seed=seed, model=name, data=test_name, subset=subset,
                                            **classification(d['target'][mask], score[mask], score[mask] > cutoff)))
                for kind, label in enumerate(KINDS):
                    mask = d['kinds'] == kind
                    pattern_rows.append(dict(seed=seed, model=name, data=test_name, pattern=label,
                                              score_mean=float(score[mask].mean()),
                                              alarm_rate=float((score[mask] > cutoff).mean()),
                                              hidden_fired_mean=float(rep[test_name][mask].sum(axis=1).mean())))
                for i in range(len(score)):
                    prediction_rows.append(dict(seed=seed, model=name, data=test_name,
                                                sample=i, kind=int(d['kinds'][i]), target=int(d['target'][i]),
                                                score=float(score[i]), cutoff=cutoff, alarm=bool(score[i] > cutoff)))
            np.savez_compressed(folder/f'{name}_scores.npz', validation=val_score,
                                clean=rep['clean']@output, noise=rep['noise']@output, cutoff=cutoff)
        h = representations['learned_delays']['train']
        target_indices = train['kinds'] == 0
        # Interpret all target-responsive cells; never use this selection for prediction.
        candidates = np.flatnonzero(h[target_indices].mean(axis=0) > .5)
        for neuron in candidates:
            order = np.argsort(-hidden.weight[neuron])[:6]
            diagnostics.append(dict(seed=seed, neuron=int(neuron), identities=identities,
                                    response_by_pattern=[float(h[train['kinds']==k,neuron].mean()) for k in range(8)],
                                    top_connections=[dict(channel=int(i//5),delay_minutes=int(i%5),
                                                          weight=float(hidden.weight[neuron,i])) for i in order]))
        dump_json(folder/'diagnostics.json', dict(active_neurons=int(np.count_nonzero(h.sum(axis=0))),
                  learned_firing_fraction=float(h.mean()),
                  random_matched_firing_fraction=float(representations['random_activity_matched']['train'].mean()),
                  weight_change_norm=float(np.linalg.norm(hidden.weight-hidden.initial_weight))))
        # Exact deterministic rerun and artifact reload on the full test arrays.
        again = CompetitivePatterns(delayed['train'].shape[1], cfg, seed).fit(delayed['train'], seed+1)
        np.testing.assert_array_equal(hidden.weight, again.weight)
        saved = np.load(folder/'hidden.npz')
        replay = (delayed['clean'] @ saved['learned'].T > cfg['threshold']).astype(float)
        np.testing.assert_array_equal(replay, representations['learned_delays']['clean'])
        for name in representations:
            output = np.load(folder/f'{name}_output.npz')['weight']
            saved_score = np.load(folder/f'{name}_scores.npz')
            np.testing.assert_array_equal(representations[name]['clean']@output, saved_score['clean'])
        print(f'seed {seed}: active hidden={np.count_nonzero(h.sum(axis=0))}, target-selective candidates={len(candidates)}',flush=True)
    pd.DataFrame(metric_rows).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(pattern_rows).to_csv(OUT/'pattern_responses.csv',index=False)
    pd.DataFrame(prediction_rows).to_parquet(OUT/'predictions.parquet',index=False)
    dump_json(OUT/'learned_patterns.json',diagnostics)
    for path, expected in hashes.items():
        assert sha256(ROOT/path) == expected,path
    dump_json(OUT/'verification.json', dict(seeds=cfg['seeds'], trained_hidden_layers=6,
              output_weight_sets=18, synthetic_train_sequences=2880,
              synthetic_validation_sequences=960, synthetic_clean_test_sequences=1920,
              noisy_test_variants=1920, exact_hidden_retraining=True,
              reloaded_output_predictions=18, training_inputs_unchanged=True,
              financial_data_used=False))
    print(pd.DataFrame(metric_rows).query("data == 'clean' and subset == 'same_AB_counts'")
          [['seed','model','auc','ap','precision','recall']].to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
