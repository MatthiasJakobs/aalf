import numpy as np
from sklearn.metrics import mean_squared_error

def rmse(a, b):
    return mean_squared_error(a, b, squared=False)

# standardized mean_squared_error
def smse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    std = np.std(y_true)
    assert std != 0, y_true
    return mse/std