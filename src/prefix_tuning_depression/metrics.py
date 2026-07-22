"""Regression metrics for depression severity prediction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(mean_absolute_error(y_true, y_pred))


def aggregate_run_results(results: list[dict[str, float]]) -> dict[str, float]:
    """Return mean and std of RMSE/MAE across multiple runs."""
    if not results:
        return {}

    rmse_values = [r["rmse"] for r in results]
    mae_values = [r["mae"] for r in results]

    return {
        "rmse_mean": float(np.mean(rmse_values)),
        "rmse_std": float(np.std(rmse_values)),
        "mae_mean": float(np.mean(mae_values)),
        "mae_std": float(np.std(mae_values)),
    }
