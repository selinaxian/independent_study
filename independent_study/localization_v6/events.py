"""Paper-style labels on minute closes; full-day pivot is evaluation-only."""
import numpy as np
import pandas as pd


def event_labels(bars,meta,windows=(3,5)):
    if not np.array_equal(bars.bar_id.to_numpy(),np.arange(len(bars))):raise ValueError('Expected contiguous bar IDs')
    close=bars.close.to_numpy();segment=bars.segment_id.to_numpy()
    change=np.full(len(bars),np.nan)
    valid=segment[:-1]==segment[1:]
    change[:-1]=np.where(valid,abs(close[1:]/close[:-1]-1)*10000,np.nan)
    pivots=pd.DataFrame({'date':bars.trading_date,'contract':bars.contract_id,'change':change}).groupby(['date','contract']).change.median()
    idx=meta.bar_id.to_numpy();out=meta[['sample_id','bar_id','trading_date','contract_id']].copy().reset_index(drop=True)
    out['day_pivot']=np.array([pivots.loc[(r.trading_date,r.contract_id)] for r in out.itertuples()])
    for window in windows:
        future=idx[:,None]+np.arange(1,window+1)[None,:]
        previous=idx[:,None]+np.arange(-window,0)[None,:]
        if future.max()+1>=len(bars) or previous.min()<0:raise ValueError('Window outside data')
        if not np.all(segment[future+1]==segment[idx,None]) or not np.all(segment[previous]==segment[idx,None]):
            raise ValueError('Event windows must stay within contract/session')
        out[f'strength_{window}']=np.mean(change[future],axis=1)
        out[f'past_abs_{window}']=np.mean(change[previous],axis=1)
        out[f'event_{window}']=(out[f'strength_{window}']>out.day_pivot).astype(np.int64)
    if not np.isfinite(out.select_dtypes('number')).all().all():raise ValueError('Nonfinite event label')
    return out


def ranking_metrics(labels,scores):
    labels=np.asarray(labels,dtype=int);scores=np.asarray(scores,dtype=float)
    positives=labels.sum();negative=len(labels)-positives
    if positives==0 or negative==0:return {'auc':np.nan,'average_precision':np.nan}
    ranks=pd.Series(scores).rank(method='average').to_numpy()
    auc=(ranks[labels==1].sum()-positives*(positives+1)/2)/(positives*negative)
    # Group tied scores before calculating recall steps; arbitrary tie order adds no skill.
    order=np.argsort(-scores,kind='stable');y=labels[order];s=scores[order]
    ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    tp=np.cumsum(y)[ends];counts=ends+1
    recall=tp/positives;ap=np.sum(np.diff(np.r_[0,recall])*(tp/counts))
    return {'auc':float(auc),'average_precision':float(ap)}


def event_summary(labels,scores,signals,dates,calendar):
    y=np.asarray(labels,dtype=int);signal=np.asarray(signals,dtype=bool)
    frame=pd.DataFrame({'date':dates,'n':1,'events':y,'signals':signal.astype(int),'hits':signal*y})
    daily=frame.groupby('date').sum().reindex(calendar,fill_value=0)
    daily['expected_hits']=np.divide(daily.events,daily.n,out=np.zeros(len(daily)),where=daily.n.to_numpy()>0)*daily.signals
    n=len(y);s=int(signal.sum());hit=int((y*signal).sum());events=int(y.sum())
    expected=float(daily.expected_hits.sum())
    return {'n':n,'events':events,'base_rate':events/n,'signals':s,'coverage':s/n,
        'precision':hit/s if s else np.nan,'recall':hit/events if events else np.nan,
        'matched_random_precision':expected/s if s else np.nan,
        'precision_lift':(hit-expected)/s if s else np.nan,**ranking_metrics(y,scores)},daily
