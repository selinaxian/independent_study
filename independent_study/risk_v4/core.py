"""Causal risk rules and accounting on a fixed, nonoverlapping opportunity clock."""
import numpy as np
import pandas as pd
from iclab.backtest import trade_accounting


def fit_risk_rule(validation_prediction, validation_dates, forecast_start, quantile=.8, floor=1.):
    dates=np.asarray(validation_dates)
    values=np.asarray(validation_prediction,dtype=float)
    if len(values)!=len(dates) or not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid validation predictions")
    if max(dates)>=forecast_start:
        raise ValueError("Risk calibration must precede the forecast")
    return {"threshold":float(np.quantile(values,quantile)),
            "median":max(float(np.median(values)),floor),"floor":floor,
            "last_calibration_date":str(max(dates)),"forecast_start":forecast_start}


def risk_lots(risk, rule, application):
    risk=np.asarray(risk,dtype=float)
    if not np.isfinite(risk).all() or (risk<0).any():
        raise ValueError("Invalid risk forecast")
    if application=="filter":return (risk<=rule["threshold"]).astype(np.int64)
    if application=="sizing":
        return np.minimum(2,np.floor(2*rule["median"]/np.maximum(risk,rule["floor"]))).astype(np.int64)
    raise ValueError(application)


def account(entry,exit,side,lots,cfg):
    if int(lots)!=lots or lots<0:raise ValueError("Integer nonnegative lots required")
    a=trade_accounting(entry,exit,side,cfg)
    if lots==0:
        return {k:(v if k.endswith("fill") else 0.) for k,v in a.items()}
    for key in ["entry_fee","exit_fee"]:
        a[key]=(a[key]-cfg["order_fee"])*lots+cfg["order_fee"]
    a["gross_pnl"]*=lots;a["slippage_cost"]*=lots
    a["net_pnl"]=a["gross_pnl"]-a["slippage_cost"]-a["entry_fee"]-a["exit_fee"]
    return a


def replay(bars,opportunities,quantities,cfg,calendar):
    """Keep all original opportunity slots, even when the overlay skips a slot.

    Quantity is an input computed at signal time. Future OHLC are used only for
    fills/marking. Margin is checked using current cash at each entry, never reset.
    """
    quantities=np.asarray(quantities)
    if len(quantities)!=len(opportunities) or (quantities<0).any() or not np.equal(quantities,np.floor(quantities)).all():
        raise ValueError("Invalid quantities")
    bars=bars.reset_index(drop=True)
    lookup={int(v):i for i,v in enumerate(bars.bar_id)}
    opens=bars.open.to_numpy();closes=bars.close.to_numpy()
    cash=float(cfg["capital"]);equity=np.empty(len(bars));last=0;records=[];rejected=0
    previous_exit=-1
    for row,lots in zip(opportunities.itertuples(),quantities.astype(int)):
        entry=lookup[int(row.entry_bar_id)];end=lookup[int(row.exit_bar_id_actual)]
        if entry<previous_exit or entry>=end:raise ValueError("Overlapping or reversed opportunities")
        previous_exit=end
        assert bars.iloc[entry].segment_id==bars.iloc[end].segment_id
        equity[last:entry]=cash
        desired=int(lots)
        if lots and cash<opens[entry]*cfg["multiplier"]*cfg["margin_fraction"]*lots+1000:
            lots=0;rejected+=1
        a=account(opens[entry],opens[end],row.side,lots,cfg)
        equity[entry:end]=cash-a["entry_fee"]+(closes[entry:end]-a["entry_fill"])*row.side*cfg["multiplier"]*lots
        cash+=a["net_pnl"];last=end
        records.append({**row._asdict(),**a,"desired_lots":desired,"lots":int(lots),"cash_after":cash})
    equity[last:]=cash
    tr=pd.DataFrame(records)
    eq=pd.DataFrame({"datetime":bars.datetime,"trading_date":bars.trading_date,"equity":equity})
    daily=eq.groupby("trading_date").equity.last().reindex(calendar).ffill().fillna(cfg["capital"])
    pnl=daily.diff().fillna(daily.iloc[0]-cfg["capital"])
    ret=pnl/daily.shift().fillna(cfg["capital"])
    path=np.r_[cfg["capital"],equity];dd=path/np.maximum.accumulate(path)-1
    active=tr.lots>0
    loss_tail=np.sort(-pnl.to_numpy())[-max(1,int(np.ceil(len(pnl)*.05))):]
    summary={"trades":int(active.sum()),"lots":int(tr.lots.sum()),"opportunities":len(tr),
             "capital":cfg["capital"],"net_pnl":cash-cfg["capital"],"return_pct":(cash/cfg["capital"]-1)*100,
             "gross_pnl":tr.gross_pnl.sum(),"fees":(tr.entry_fee+tr.exit_fee).sum(),"slippage":tr.slippage_cost.sum(),
             "max_drawdown_pct":dd.min()*100,"daily_pnl_std":pnl.std(ddof=1),
             "daily_sharpe":float(ret.mean()/ret.std(ddof=1)*np.sqrt(252)) if ret.std(ddof=1)>1e-12 else 0.,
             "worst_day":pnl.min(),"daily_loss_es95":loss_tail.mean(),"margin_rejections":rejected,
             "average_net_per_lot":tr.net_pnl.sum()/max(1,tr.lots.sum())}
    np.testing.assert_allclose(tr.net_pnl.sum(),summary["net_pnl"],atol=1e-6,rtol=0)
    return summary,tr,eq,pd.DataFrame({"trading_date":calendar,"equity":daily.values,"pnl":pnl.values})


def matched_exposure(trades,cfg,repetitions=1000,seed=20261028):
    """Post-hoc attribution only: permute lot assignments inside each fixed fold."""
    rng=np.random.default_rng(seed);draws=np.zeros(repetitions);expected=np.zeros(len(trades))
    # Zero-lot records still have observed opportunity prices, never used by policy.
    unit=np.array([trade_accounting(r.entry_open,r.exit_open,r.side,cfg)["net_pnl"]
                   for r in trades.itertuples()])
    # Fixed fees are per order, not per contract.
    variable=unit+2*cfg["order_fee"]
    for _,part in trades.groupby("fold",sort=True):
        idx=part.index.to_numpy();q=part.lots.to_numpy()
        expected[idx]=q.mean()*variable[idx]-2*cfg["order_fee"]*np.mean(q>0)
        for b in range(repetitions):
            perm=rng.permutation(q)
            draws[b]+=np.dot(perm,variable[idx])-2*cfg["order_fee"]*np.count_nonzero(perm)
    actual=float(trades.net_pnl.sum())
    return {"matched_expected_net":float(expected.sum()),"selection_excess":actual-expected.sum(),
            "random_net_low":float(np.quantile(draws,.025)),"random_net_high":float(np.quantile(draws,.975)),
            "random_percentile":float(np.mean(draws<=actual)),
            "control_ignores_margin":True},expected


def block_interval(daily_difference,repetitions=1000,block=5,seed=20261028):
    a=np.asarray(daily_difference,dtype=float);n=len(a)
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,max(1,n-block+1),(repetitions,int(np.ceil(n/block))))
    indices=(starts[:,:,None]+np.arange(min(block,n))).reshape(repetitions,-1)[:,:n]
    totals=a[indices].sum(axis=1)
    return float(np.quantile(totals,.025)),float(np.quantile(totals,.975))
