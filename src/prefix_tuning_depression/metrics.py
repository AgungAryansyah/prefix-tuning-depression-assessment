"""Regression metrics for depression severity prediction."""

from __future__ import annotations

import numpy as np
from scipy.stats import f_oneway
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


def pairwise_error_anova(
    labels: np.ndarray,
    reference_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compare two models' absolute and squared errors with one-way ANOVA."""
    labels = np.asarray(labels)
    reference_predictions = np.asarray(reference_predictions)
    candidate_predictions = np.asarray(candidate_predictions)
    if (
        labels.shape != reference_predictions.shape
        or labels.shape != candidate_predictions.shape
    ):
        raise ValueError("labels and predictions must have identical shapes")

    reference_errors = labels - reference_predictions
    candidate_errors = labels - candidate_predictions
    return {
        "absolute": _anova(np.abs(reference_errors), np.abs(candidate_errors)),
        "squared": _anova(reference_errors**2, candidate_errors**2),
    }


def _anova(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    result = f_oneway(first, second)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }
