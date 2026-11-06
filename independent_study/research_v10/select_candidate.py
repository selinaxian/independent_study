"""Reproduce the development-only economic candidate choice, without overwrites."""
import json
import numpy as np
import pandas as pd
from iclab.data import dump_json,sha256
from research_v10.state_run import ROOT


def main():
    out=ROOT/'independent_study/results/v10_refit';path=out/'economic_selection_lock.json'
    frame=pd.read_csv(out/'development_metrics.csv').set_index('model')
    names=['old40_stdp','gated40_stdp','readout50_stdp','full50_stdp']
    scores={m:float(frame.loc[m,'qlike']) for m in names};primary=min(scores,key=scores.get)
    if path.exists():
        prior=json.loads(path.read_text());assert prior['primary']==primary
        for m,value in scores.items():np.testing.assert_allclose(value,prior['development_scores'][m],rtol=0,atol=1e-14)
        print('Existing selection verified, unchanged:',primary);return
    if (out/'audit_metrics.csv').exists():raise RuntimeError('Cannot create a prospective selection lock after audit results exist')
    dump_json(path,dict(primary=primary,criterion='Lowest development QLIKE among prespecified STDP framework candidates; not selected by trading PnL',
        development_scores=scores,chosen_before_refit_audit_metrics=True,development_metrics_sha256=sha256(out/'development_metrics.csv')))
    print('Locked development-only candidate:',primary)


if __name__=='__main__':main()
