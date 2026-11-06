import numpy as np
import pandas as pd


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x)<3 or np.std(x)<1e-12 or np.std(y)<1e-12: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def fit_readout(train, val, yt, yv, alphas):
    mean, scale = train.mean(axis=0), train.std(axis=0)
    scale = np.maximum(scale, 1e-6)
    a, b = (train-mean)/scale, (val-mean)/scale
    ym = yt.mean(axis=0)
    eig, vec = np.linalg.eigh(a.T@a)
    rhs = vec.T@(a.T@(yt-ym))
    coefs = np.array([vec@(rhs/(np.maximum(eig, 0)[:, None]+alpha)) for alpha in alphas])
    candidates = np.einsum("nd,adj->anj", b, coefs)+ym
    # Absolute-return readout is nonnegative in every comparison.
    candidates[:, :, -1] = np.maximum(candidates[:, :, -1], 0)
    mse = ((candidates-yv)**2).mean(axis=1)
    best = mse.argmin(axis=0)
    coef = np.column_stack([coefs[best[j], :, j] for j in range(yt.shape[1])])
    return {"mean": mean, "scale": scale, "target_mean": ym, "coef": coef, "alphas": np.asarray(alphas)[best], "validation_mse_grid": mse}


def predict_readout(model, x):
    p = ((x-model["mean"])/model["scale"])@model["coef"]+model["target_mean"]
    p[:, -1] = np.maximum(p[:, -1], 0)
    return p


def evaluate(pred, target, dates, risk=False):
    p, y = np.asarray(pred), np.asarray(target)
    daily = pd.DataFrame({"date": dates, "p": p, "y": y}).groupby("date")
    daily_ic = [correlation(g.p, g.y) for _, g in daily]
    finite = np.asarray(daily_ic)[np.isfinite(daily_ic)]
    result = {"n": len(p), "ic": correlation(p, y), "rank_ic": correlation(pd.Series(p).rank(), pd.Series(y).rank()), "rmse": float(np.sqrt(np.mean((p-y)**2))), "mae": float(np.mean(np.abs(p-y))), "daily_ic_mean": float(finite.mean()) if len(finite) else float("nan"), "positive_ic_days": float((finite>0).mean()) if len(finite) else float("nan")}
    if not risk: result["direction_accuracy"] = float(np.mean(np.sign(p)==np.sign(y)))
    return result


def sufficient_daily(p, y, dates, calendar_dates):
    df = pd.DataFrame({"date": dates, "n": 1., "x": p, "y": y, "xx": p*p, "yy": y*y, "xy": p*y})
    return df.groupby("date").sum().reindex(calendar_dates, fill_value=0).to_numpy()


def corr_sums(s):
    n, x, y, xx, yy, xy = np.asarray(s).T
    denom = np.sqrt(np.maximum(xx-x*x/np.maximum(n,1), 0)*np.maximum(yy-y*y/np.maximum(n,1), 0))
    return np.divide(xy-x*y/np.maximum(n,1), denom, out=np.full_like(denom, np.nan), where=denom>1e-15)


def block_bootstrap(preds, y, dates, calendar_dates, reps=1000, block=5):
    rng = np.random.default_rng(62319)
    n = len(calendar_dates)
    starts = rng.integers(0, max(n-block+1, 1), (reps, int(np.ceil(n/block))))
    indices = ((starts[:, :, None]+np.arange(block)) % n).reshape(reps, -1)[:, :n]
    distributions, intervals = {}, {}
    for name, p in preds.items():
        sums = sufficient_daily(p, y, dates, calendar_dates)
        d = corr_sums(sums[indices].sum(axis=1))
        distributions[name] = d
        intervals[name] = {"ic": correlation(p, y), "ci_low": float(np.nanquantile(d,.025)), "ci_high": float(np.nanquantile(d,.975))}
    for name in ("stdp", "raw_plus_stdp", "random", "shuffle"):
        if name in distributions and "raw_history" in distributions:
            d = distributions[name]-distributions["raw_history"]
            intervals[name+"_minus_raw_history"] = {"ic": intervals[name]["ic"]-intervals["raw_history"]["ic"], "ci_low": float(np.nanquantile(d,.025)), "ci_high": float(np.nanquantile(d,.975))}
    return intervals
