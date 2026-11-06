import numpy as np
import pandas as pd
from risk_v4.core import matched_exposure


def safe_corr(left, right, rank=False):
    a,b=pd.Series(left,dtype=float),pd.Series(right,dtype=float)
    if len(a)<2 or a.nunique()<2 or b.nunique()<2:
        return np.nan
    if rank:a,b=a.rank(),b.rank()
    return a.corr(b)


def covered_matched_exposure(trades, cfg, repetitions=1000, seed=20261028):
    """Preserve fold AND causal applicability, not merely total exposure."""
    copy=trades.copy()
    copy['fold']=copy.fold.astype(int)*2+copy.covered.astype(int)
    return matched_exposure(copy,cfg,repetitions,seed)
