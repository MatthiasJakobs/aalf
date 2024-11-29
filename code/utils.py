import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from os.path import exists

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

def load_dict_or_create(path):
    if not exists(path):
        return {}
    else:
        with open(path, 'rb') as f:
            return pickle.load(f)
            
# Highlight the minimum value in each row
def highlight_min_multicolumn(row, metric_names=None):
    formatted_row = row.copy()
    for metric_name in metric_names:
        best_model = row[:, metric_name].idxmin()
        formatted_row[best_model, metric_name] = fr'\textbf{{{row[best_model, metric_name]}}}'
    return formatted_row

def format_significant(row):
    return row.apply(lambda x: f"{x:.3f}")