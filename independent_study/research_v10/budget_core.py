import numpy as np


def budget_lots(variance,known_close,budget,multiplier=200.):
    variance=np.asarray(variance,float);close=np.asarray(known_close,float)
    if not np.isfinite(variance).all() or (variance<=0).any() or (close<=0).any():
        raise ValueError('Finite positive variance/price required')
    unit_risk=close*multiplier*np.sqrt(variance)/10000
    return np.minimum(2,np.floor(np.asarray(budget)/unit_risk)).astype(int),unit_risk


def matched_risk(trades,unit_risk,budget):
    """Exact conditional expectation under fixed fold/contract/session counts."""
    expected=np.zeros(len(trades));exceed=np.zeros(len(trades))
    for _,g in trades.groupby(['fold','contract_id','session'],sort=True):
        ix=g.index.to_numpy();q=g.lots.to_numpy()
        expected[ix]=np.mean(q*q)*unit_risk[ix]**2
        for lots in [0,1,2]:exceed[ix]+=np.mean(q==lots)*(lots*unit_risk[ix]>budget[ix])
    return expected,exceed
