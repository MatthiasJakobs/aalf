import numpy as np
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError

def mae(a, b):
    return mean_absolute_error(a, b)

def mse(a, b):
    return mean_squared_error(a, b)

def rmse(a, b):
    if isinstance(a, np.ndarray):
        return root_mean_squared_error(a, b)

    return ((a - b)**2).mean().sqrt()

def smape(y_true, y_pred):
    return MeanAbsolutePercentageError(symmetric=True)(y_true, y_pred)