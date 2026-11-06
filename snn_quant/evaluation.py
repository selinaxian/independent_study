"""Train-only ridge fits; validation-only regularization choice; frozen test."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RidgeReadout:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    alpha: float

    def predict(self, x):
        return ((np.asarray(x, dtype=np.float64) - self.mean) / self.scale
                @ self.coefficient + self.intercept)

    def save(self, path):
        np.savez_compressed(path, mean=self.mean, scale=self.scale,
                            coefficient=self.coefficient, intercept=self.intercept, alpha=self.alpha)


def select_ridge(x_train, y_train, x_validation, y_validation, alphas):
    x_train = np.asarray(x_train, dtype=np.float64)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x_train - mean) / scale
    intercept = float(np.mean(y_train))
    covariance = z.T @ z / len(z)
    values, vectors = np.linalg.eigh(covariance)
    values = values.clip(min=0)
    projection = vectors.T @ (z.T @ (y_train - intercept) / len(z))
    candidates, models = [], []
    for alpha in alphas:
        if alpha <= 0:
            raise ValueError("Ridge regularization must be positive")
        coefficient = vectors @ (projection / (values + alpha))
        model = RidgeReadout(mean, scale, coefficient, intercept, float(alpha))
        prediction = model.predict(x_validation)
        mse = float(np.mean((prediction - y_validation) ** 2))
        models.append(model)
        candidates.append({"alpha": float(alpha), "validation_mse_bps2": mse})
    selected = int(np.argmin([v["validation_mse_bps2"] for v in candidates]))
    return models[selected], candidates


def correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(y, prediction, dates):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Invalid evaluation values")
    frame = pd.DataFrame({"y": y, "prediction": prediction, "date": np.asarray(dates)})
    daily = {str(date): correlation(group.y, group.prediction)
             for date, group in frame.groupby("date", sort=True)}
    daily_values = [v for v in daily.values() if v is not None]
    nonzero = y != 0
    return {
        "n": len(y),
        "dates": int(frame.date.nunique()),
        "rmse_bps": float(np.sqrt(np.mean((prediction - y) ** 2))),
        "mae_bps": float(np.mean(np.abs(prediction - y))),
        "pearson_ic": correlation(y, prediction),
        "spearman_ic": correlation(pd.Series(y).rank(), pd.Series(prediction).rank()),
        "direction_accuracy": float(np.mean(np.sign(prediction[nonzero]) == np.sign(y[nonzero])))
        if nonzero.any() else None,
        "nonzero_target_count": int(nonzero.sum()),
        "daily_ic_mean": float(np.mean(daily_values)) if daily_values else None,
        "daily_ic": daily,
    }
