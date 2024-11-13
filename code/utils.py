import numpy as np
from sklearn.metrics import mean_squared_error, root_mean_squared_error

def rmse(a, b):
    if isinstance(a, np.ndarray):
        return root_mean_squared_error(a, b)

    return ((a - b)**2).mean().sqrt()

# standardized mean_squared_error
def smse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    std = np.std(y_true)
    assert std != 0, y_true
    return mse/std