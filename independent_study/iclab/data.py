from pathlib import Path
import calendar
import hashlib
import json
import numpy as np
import pandas as pd

FEATURES = ["return_1_atr", "return_3_atr", "efficiency_5", "resistance_distance", "support_distance", "close_location", "volatility_ratio", "range_ratio", "volume_surprise", "oi_change", "volatility_level", "session_progress"]
TARGETS = ["return_1", "return_3", "return_5", "return_10", "absolute_5"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert), encoding="utf-8")


def expiry_cutoff(contract, days):
    year, month = 2000 + int(contract[2:4]), int(contract[4:6])
    fridays = [w[calendar.FRIDAY] for w in calendar.monthcalendar(year, month) if w[calendar.FRIDAY]]
    return (pd.Timestamp(year, month, fridays[2]) - pd.Timedelta(days=days)).strftime("%Y-%m-%d")


def select_contracts(bars, dates, cfg):
    daily = bars.groupby(["trading_date", "contract_id"], sort=True).agg(rows=("close", "size"), volume=("volume", "sum"), oi=("open_interest", "last")).reset_index()
    decisions, chosen = [], []
    for k, day in enumerate(dates):
        previous = dates[k-1] if k else None
        r = {"trading_date": day, "information_date": previous, "contract_id": None, "status": "no_previous_data"}
        prior = daily[daily.trading_date.eq(previous)].copy()
        if not prior.empty:
            prior = prior[(prior.rows == 240) & (prior.volume >= cfg["min_previous_volume"]) & (prior.oi >= cfg["min_previous_oi"])]
            prior = prior[[day < expiry_cutoff(c, cfg["avoid_expiry_days"]) for c in prior.contract_id]]
            if prior.empty:
                r["status"] = "no_eligible_contract"
            else:
                best = prior.sort_values(["oi", "volume", "contract_id"], ascending=[False, False, True]).iloc[0]
                r.update(contract_id=best.contract_id, previous_volume=int(best.volume), previous_oi=int(best.oi))
                current = bars[bars.trading_date.eq(day) & bars.contract_id.eq(best.contract_id)]
                if len(current) != 240:
                    r["status"] = "selected_contract_missing_or_incomplete"
                else:
                    r["status"] = "selected"
                    chosen.append(current)
        decisions.append(r)
    selected = pd.concat(chosen).sort_values("datetime").reset_index(drop=True) if chosen else bars.iloc[:0].copy()
    return selected, pd.DataFrame(decisions)


def feature_segment(g, lookback=20):
    c, h, l, v, oi = [g[n].astype(float) for n in ["close", "high", "low", "volume", "open_interest"]]
    prev = c.shift(1)
    tr = pd.concat([h-l, (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    tr.iloc[0] = np.nan
    atr = tr.shift(1).rolling(lookback).mean().clip(lower=1e-6)
    r = np.log(c).diff()
    vol20 = r.shift(1).rolling(lookback).std(ddof=0)
    eps = 1e-8
    f = pd.DataFrame(index=g.index)
    f[FEATURES[0]] = c.diff() / atr
    f[FEATURES[1]] = c.diff(3) / atr
    f[FEATURES[2]] = c.diff(5) / c.diff().abs().rolling(5).sum().clip(lower=eps)
    f[FEATURES[3]] = (h.shift(1).rolling(lookback).max() - c) / atr
    f[FEATURES[4]] = (c - l.shift(1).rolling(lookback).min()) / atr
    f[FEATURES[5]] = (2*c-h-l) / (h-l).clip(lower=1e-6)
    f[FEATURES[6]] = np.log((r.rolling(5).std(ddof=0)+eps)/(vol20+eps))
    f[FEATURES[7]] = np.log(((h-l).clip(lower=1e-6))/atr)
    f[FEATURES[8]] = np.log1p(v) - np.log1p(v.shift(1).rolling(lookback).mean())
    f[FEATURES[9]] = np.log1p(oi).diff()
    f[FEATURES[10]] = np.log1p(vol20 * 10000)
    # Known clock position, not the observed number of future rows.
    minute = g.datetime.dt.hour*60 + g.datetime.dt.minute
    start = np.where(minute < 12*60, 9*60+31, 13*60+1)
    f[FEATURES[11]] = 2*(minute-start)/119 - 1
    return f.replace([np.inf, -np.inf], np.nan)


def prepare(cfg, out):
    root = Path(cfg["data_dir"])
    files = sorted((root/"contracts").glob("*.parquet"))
    bars = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True).sort_values(["contract_id", "datetime"])
    assert not bars.duplicated(["contract_id", "datetime"]).any()
    assert bars.contract_id.notna().all()
    dates = pd.read_csv(root/"metadata/archives.csv")["archive_date"].astype(str)
    dates = sorted(pd.to_datetime(dates, format="%Y%m%d").dt.strftime("%Y-%m-%d").unique())
    dates = [d for d in dates if d <= cfg["test_end"]]
    selected, decisions = select_contracts(bars, dates, cfg["selection"])
    selected["bar_id"] = np.arange(len(selected))
    boundary = selected.contract_id.ne(selected.contract_id.shift()) | selected.trading_date.ne(selected.trading_date.shift()) | selected.session.ne(selected.session.shift()) | selected.datetime.diff().ne(pd.Timedelta(minutes=1))
    selected["segment_id"] = boundary.cumsum().astype(int)
    selected["split"] = np.select([selected.trading_date <= cfg["train_end"], selected.trading_date <= cfg["validation_end"]], ["train", "validation"], "test")
    features = pd.concat([feature_segment(g, cfg["feature_lookback"]) for _, g in selected.groupby("segment_id")]).sort_index()
    selected = pd.concat([selected, features], axis=1)
    fit = selected.loc[selected.split.eq("train"), FEATURES].dropna()
    scale = np.maximum(np.quantile(np.abs(fit.to_numpy()), cfg["scale_quantile"], axis=0), 1e-8)
    normalized = np.clip(selected[FEATURES].to_numpy()/scale, -1, 1).astype(np.float32)
    windows, labels, meta = [], [], []
    length, offset = cfg["window_minutes"], cfg["entry_bar_offset"]
    for sid, g in selected.groupby("segment_id", sort=True):
        indices = g.index.to_numpy()
        for j in range(length-1, len(g)-offset-max(cfg["horizons"])-1):
            i = indices[j]
            window = normalized[indices[j-length+1:j+1]]
            if not np.isfinite(window).all(): continue
            entry = g.iloc[j+offset]
            returns = [(g.iloc[j+offset+h].open/entry.open-1)*10000 for h in cfg["horizons"]]
            row = selected.loc[i]
            windows.append(window)
            labels.append(returns + [abs(returns[2])])
            meta.append({"sample_id": len(meta), "bar_id": int(i), "segment_id": int(sid), "contract_id": row.contract_id, "trading_date": row.trading_date, "datetime": row.datetime, "window_start": selected.loc[indices[j-length+1], "datetime"], "split": row.split, "entry_bar_id": int(entry.bar_id), "exit_bar_id": int(g.iloc[j+offset+5].bar_id), "signal_volume": int(row.volume)})
    X, y, meta = np.asarray(windows), np.asarray(labels, dtype=np.float64), pd.DataFrame(meta)
    out.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(out/"bars_features.parquet", index=False)
    decisions.to_csv(out/"contract_selection.csv", index=False)
    meta.to_parquet(out/"samples.parquet", index=False)
    np.savez_compressed(out/"dataset.npz", X=X, y=y, scale=scale)
    summary = {"features": FEATURES, "targets": TARGETS, "scale": scale, "scale_fit_minutes": len(fit), "raw_contract_rows": len(bars), "selected_rows": len(selected), "selected_days": int(decisions.status.eq("selected").sum()), "calendar_days": len(dates), "selection_status": decisions.status.value_counts().to_dict(), "samples": meta.groupby("split").size().to_dict(), "sample_days": meta.groupby("split").trading_date.nunique().to_dict(), "sample_contracts": meta.groupby(["split", "contract_id"]).size().to_dict(), "input_hashes": {str(p): sha256(p) for p in files + [root/"metadata/archives.csv"]}}
    summary["sample_contracts"] = {"/".join(k): v for k, v in summary["sample_contracts"].items()}
    dump_json(out/"data_summary.json", summary)
    return X, y, meta, selected, decisions, summary
