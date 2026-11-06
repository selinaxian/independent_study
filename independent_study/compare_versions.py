"""Post-hoc common-date comparison of saved forecasts; no fitting or selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from iclab.analysis import correlation,sufficient_daily,corr_sums
from iclab.data import sha256
from diagnostics_v2.common import save_json


def main():
    root=Path('independent_study/results');out=Path('outputs/version_comparison_20261028');out.mkdir(exist_ok=True)
    files=[root/'v1/test_predictions.parquet',root/'v1/contract_selection.csv',root/'v2/forecast_samples.parquet',root/'v2/ensemble_predictions.npz',root/'v2/dataset.npz',root/'v3/forecast_samples.parquet',root/'v3/ensemble_predictions.npz']
    v1=pd.read_parquet(files[0]);ids=v1.sample_id.to_numpy();dates=v1.trading_date.to_numpy()
    m2=pd.read_parquet(files[2]);m3=pd.read_parquet(files[5])
    i2=pd.Index(m2.sample_id).get_indexer(ids);i3=pd.Index(m3.sample_id).get_indexer(ids)
    assert len(ids)==6900 and (i2>=0).all() and (i3>=0).all()
    for m,ix in [(m2,i2),(m3,i3)]:
        for col in ['datetime','contract_id','bar_id','entry_bar_id','exit_bar_id']:
            np.testing.assert_array_equal(v1[col].to_numpy(),m.iloc[ix][col].to_numpy())
    y=v1[['actual_return_5','actual_absolute_5']].to_numpy()
    y2=np.load(files[4])['y'][ids,:2]
    np.testing.assert_allclose(y,y2,rtol=0,atol=1e-5)
    predictions={'v1_stdp':v1[['stdp__return_5','stdp__absolute_5']].to_numpy(),
                 'v1_raw':v1[['raw_history__return_5','raw_history__absolute_5']].to_numpy()}
    with np.load(files[3]) as a:
        for arm in ['frozen','readout_only','rolling_all']:
            predictions['v2_'+arm+'_stdp']=a[arm+'__stdp__linear'][i2,:2]
        predictions['v2_rolling_all_raw']=a['rolling_all__raw_history__linear'][i2,:2]
    with np.load(files[6]) as a:
        for model in ['soft_fixed_stdp','soft_adapt_stdp','deep_last_stdp','deep_concat_stdp']:
            predictions['v3_'+model]=a[model+'__linear'][i3,:2]
    calendar=pd.read_csv(files[1]).trading_date
    calendar=calendar[(calendar>=dates.min())&(calendar<=dates.max())].tolist()
    n=len(calendar);rng=np.random.default_rng(92231);starts=rng.integers(0,n-4,(1000,int(np.ceil(n/5))))
    draws=(starts[:,:,None]+np.arange(5)).reshape(1000,-1)[:,:n]
    metrics=[];distributions={};points={}
    for model,p in predictions.items():
        distributions[model]={};points[model]={}
        for j,target in enumerate(['return_5','absolute_5']):
            point=correlation(p[:,j],y[:,j]);s=sufficient_daily(p[:,j],y[:,j],dates,calendar)
            d=corr_sums(s[draws].sum(1));finite=d[np.isfinite(d)]
            distributions[model][target]=d;points[model][target]=point
            metrics.append({'model':model,'target':target,'n':len(ids),'ic':point,'ci_low':np.quantile(finite,.025),'ci_high':np.quantile(finite,.975)})
    differences=[]
    for model in predictions:
        if model in ['v1_stdp','v1_raw','v2_rolling_all_raw']:continue
        references=['v1_stdp']+(['v2_rolling_all_stdp'] if model.startswith('v3_') else [])
        for ref in references:
            for target in ['return_5','absolute_5']:
                d=distributions[model][target]-distributions[ref][target];d=d[np.isfinite(d)]
                differences.append({'model':model,'reference':ref,'target':target,'difference':points[model][target]-points[ref][target],'ci_low':np.quantile(d,.025),'ci_high':np.quantile(d,.975)})
    metrics=pd.DataFrame(metrics);metrics.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(differences).to_csv(out/'paired_comparisons.csv',index=False)
    save_json(out/'audit.json',{'purpose':'Post-hoc descriptive comparison of saved predictions, no training, tuning or model selection',
        'first_date':dates.min(),'last_date':dates.max(),'rows':len(ids),'actual_days':len(set(dates)),'package_days':len(calendar),
        'metadata_exact_match':True,'target_max_abs_difference_bps':float(abs(y-y2).max()),
        'bootstrap_repetitions':1000,'block_package_days':5,
        'limitations':'Evaluation dates match; original training/validation histories still differ. This does not isolate rolling causally or create a new unseen test.',
        'sources':{str(p):sha256(p) for p in files+[Path(__file__)]}})
    print(metrics.pivot(index='model',columns='target',values='ic').to_string())
    print(pd.DataFrame(differences).to_string(index=False))


if __name__=='__main__':main()
