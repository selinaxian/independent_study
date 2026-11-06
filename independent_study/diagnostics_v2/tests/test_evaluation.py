import json
import numpy as np
import pandas as pd
from iclab.data import sha256
from diagnostics_v2.common import classification
from analyze_v2 import confusion_daily,balanced_accuracy,load_predictions


def test_daily_confusion_preserves_missing_days_and_matches_metric():
    y=np.array([0,1,2,3,0,1,2,3])
    guessed=np.array([0,2,2,3,0,1,0,3]);scores=np.eye(4)[guessed]
    dates=np.array(["a"]*4+["c"]*4)
    daily=confusion_daily(scores,y,dates,["a","b","c"])
    assert daily[1].sum()==0 and daily.sum()==8
    expected=classification(y,scores)["balanced_accuracy"]
    assert balanced_accuracy(daily.sum(axis=0)[None,:])[0]==expected
    np.testing.assert_array_equal(balanced_accuracy(np.stack([daily.sum(axis=0)]*3)),[expected]*3)


def test_forecast_loader_preserves_date_order_and_rejects_modified_files(tmp_path):
    import pytest
    for fold,day in [(0,"2026-04-01"),(1,"2026-04-02")]:
        d=tmp_path/"folds"/f"fold_{fold:02d}";d.mkdir(parents=True)
        meta=pd.DataFrame({"sample_id":[2*fold,2*fold+1],"datetime":pd.date_range(day+" 10:00",periods=2,freq="min",tz="Asia/Shanghai")})
        meta.to_parquet(d/"samples.parquet",index=False)
        np.savez(d/"predictions.npz",frozen__stdp__linear=np.full((2,11),fold))
        (d/"complete.json").write_text(json.dumps({"predictions_sha256":sha256(d/"predictions.npz"),"locks":{}}))
    meta,a=load_predictions(tmp_path)
    np.testing.assert_array_equal(meta.sample_id,[0,1,2,3])
    np.testing.assert_array_equal(a["frozen__stdp__linear"][:,0],[0,0,1,1])
    np.savez(tmp_path/"folds/fold_01/predictions.npz",frozen__stdp__linear=np.ones((2,11))*9)
    with pytest.raises(AssertionError):load_predictions(tmp_path)
