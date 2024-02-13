import numpy as np
from sklearn.metrics import mean_squared_error
from tsx.model_selection import ROC_Member
from tsx.models.forecaster import split_zero

def rmse(a, b):
    return mean_squared_error(a, b, squared=False)

def compute_rocs(x, y, explanations, errors, threshold=0):
    # Threshold and invert explanations 
    explanations =  explanations / explanations.sum(axis=1).reshape(-1, 1)
    explanations = -explanations * ((-explanations) > threshold)

    rocs = []
    if len(x) == 0:
        return rocs
    for i, e in enumerate(explanations):
        splits = split_zero(e, min_size=3)
        for (f, t) in splits:
            r = ROC_Member(x[i], y[i], np.arange(t-f+1)+f, errors[i])
            rocs.append(r)

    return rocs
