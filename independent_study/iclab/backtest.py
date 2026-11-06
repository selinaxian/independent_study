import numpy as np
import pandas as pd


def trade_accounting(entry, exit, side, cfg):
    slip = cfg["tick"]*cfg["slippage_ticks"]
    entry_fill, exit_fill = entry+side*slip, exit-side*slip
    mult = cfg["multiplier"]
    entry_fee = entry_fill*mult*(cfg["open_fee_rate"]+cfg["broker_extra_rate"])+cfg["order_fee"]
    exit_fee = exit_fill*mult*(cfg["close_today_fee_rate"]+cfg["broker_extra_rate"])+cfg["order_fee"]
    gross = (exit-entry)*side*mult
    return {"entry_fill": entry_fill, "exit_fill": exit_fill, "entry_fee": entry_fee, "exit_fee": exit_fee, "gross_pnl": gross, "slippage_cost": 2*slip*mult, "net_pnl": gross-2*slip*mult-entry_fee-exit_fee}


def simulate(bars, meta, scores, threshold, cfg, calendar_dates, extra_delay=0, enabled=True):
    # Event clock: bar open (exit / entry) -> bar close (mark / new signal).
    signal_map = {int(m.bar_id): (m, float(p)) for m, p in zip(meta.itertuples(), scores)}
    available = {int(r.bar_id): r for r in bars.itertuples()}
    cash = float(cfg["capital"])
    position, pending = None, None
    trades, equity = [], []
    rejected = {"margin": 0, "zero_volume_entry": 0, "exit_zero_volume_delays": 0}
    for row in bars.itertuples():
        bid = int(row.bar_id)
        if position is not None and bid >= position["exit_bar_id"]:
            if row.volume > 0:
                acc = trade_accounting(position["entry_open"], row.open, position["side"], cfg)
                cash += (acc["exit_fill"]-acc["entry_fill"])*position["side"]*cfg["multiplier"]-acc["exit_fee"]
                trade = {**position, **acc, "exit_bar_id_actual": bid, "exit_time": row.datetime-pd.Timedelta(minutes=1), "exit_open": row.open, "cash_after": cash}
                trades.append(trade)
                position = None
            else:
                rejected["exit_zero_volume_delays"] += 1
        if pending is not None and bid == pending["entry_bar_id"]:
            if row.volume <= 0:
                rejected["zero_volume_entry"] += 1
            elif cash < row.open*cfg["multiplier"]*cfg["margin_fraction"]+1000:
                rejected["margin"] += 1
            else:
                position = {**pending, "entry_open": row.open, "entry_time": row.datetime-pd.Timedelta(minutes=1)}
                a = trade_accounting(row.open, row.open, position["side"], cfg)
                position["entry_fill"] = a["entry_fill"]
                cash -= a["entry_fee"]
            pending = None
        # If a delayed zero-volume exit cannot execute before a segment ends,
        # fail explicitly rather than inventing a close fill or silently carrying.
        nxt = available.get(bid+1)
        if position is not None and (nxt is None or nxt.segment_id != row.segment_id):
            raise RuntimeError("Unfillable exit at segment boundary")
        eq = cash if position is None else cash+(row.close-position["entry_fill"])*position["side"]*cfg["multiplier"]
        equity.append({"datetime": row.datetime, "trading_date": row.trading_date, "equity": eq, "cash": cash, "position": 0 if position is None else position["side"]})
        if enabled and position is None and pending is None and bid in signal_map:
            m, score = signal_map[bid]
            if abs(score) < threshold or score == 0 or row.volume < cfg["min_signal_volume"]: continue
            entry_id = int(m.entry_bar_id)+extra_delay
            exit_id = entry_id+cfg["holding_minutes"]
            if entry_id not in available or exit_id not in available: continue
            if available[exit_id].segment_id != row.segment_id: continue
            pending = {"sample_id": int(m.sample_id), "signal_bar_id": bid, "signal_time": row.datetime, "trading_date": row.trading_date, "contract_id": row.contract_id, "side": 1 if score>0 else -1, "score_bps": score, "entry_bar_id": entry_id, "exit_bar_id": exit_id, "signal_volume": int(row.volume)}
    if position is not None or pending is not None: raise RuntimeError("Position/pending order left at end")
    tr = pd.DataFrame(trades)
    eq = pd.DataFrame(equity)
    daily = eq.groupby("trading_date").equity.last().reindex(calendar_dates).ffill().fillna(cfg["capital"])
    previous = daily.shift(1).fillna(cfg["capital"])
    dr = daily/previous-1
    peak = np.maximum.accumulate(np.r_[cfg["capital"], eq.equity.to_numpy()])
    dd = np.r_[cfg["capital"], eq.equity.to_numpy()]/peak-1
    net = float(daily.iloc[-1]-cfg["capital"])
    summary = {"trades": len(tr), "gross_pnl": float(tr.gross_pnl.sum()) if len(tr) else 0., "net_pnl": net, "return_pct": net/cfg["capital"]*100, "fees": float((tr.entry_fee+tr.exit_fee).sum()) if len(tr) else 0., "slippage": float(tr.slippage_cost.sum()) if len(tr) else 0., "max_drawdown_pct": float(dd.min()*100), "daily_sharpe": float(dr.mean()/dr.std(ddof=1)*np.sqrt(252)) if dr.std(ddof=1)>1e-12 else 0., "win_rate": float((tr.net_pnl>0).mean()) if len(tr) else None, "average_net": float(tr.net_pnl.mean()) if len(tr) else 0., "calendar_days": len(daily), "active_days": int(tr.trading_date.nunique()) if len(tr) else 0, **rejected}
    if len(tr):
        assert abs(tr.net_pnl.sum()-net)<1e-6
        assert (tr.entry_time>tr.signal_time).all()
        assert tr.contract_id.notna().all()
    return summary, tr, eq, pd.DataFrame({"trading_date": daily.index, "equity": daily.values, "return": dr.values})
