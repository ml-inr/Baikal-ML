import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
)


def binary_clf_metrics(y_pred, y_true):
    y_pred = np.array(y_pred, dtype=np.int32)
    y_true = np.array(y_true, dtype=np.int32)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def regression_metrics(y_pred, y_true):
    y_pred = np.array(y_pred, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.float32)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }


def regression_and_clf_metrics(y_pred, y_true):
    metrics = {}
    print(y_pred.shape, y_true.shape)
    metrics.update(
        binary_clf_metrics(y_pred[:, 0].reshape(-1), y_true[:, 0].reshape(-1))
    )
    metrics.update(
        binary_clf_metrics(y_pred[:, 1].reshape(-1), y_true[:, 1].reshape(-1))
    )
    return metrics
