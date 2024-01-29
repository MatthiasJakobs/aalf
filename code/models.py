import torch
import torch.nn as nn
import numpy as np

from scipy.stats import mode
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class ThreeSubsetEnsemble:

    def __init__(self, alpha, random_state=None):
        self.alpha = alpha
        self.rng = np.random.RandomState(random_state)

    def _get_epsilon(self, p_one, N):
        r = min(p_one / (1-p_one), (1-p_one)/p_one)
        expr = np.exp(((-self.alpha**2 * (N/2)) / (r + self.alpha * 1/3)) + np.log(2))
        return r, expr

    def fit(self, X, y):
        T = 1

        # Split data initial
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=self.rng)

        emp_one = np.mean(y_val)




class DownsampleEnsembleClassifier:

    def __init__(self, model_class, n_member, *args, random_state=None, **kwargs):
        self.n_member = n_member
        self.rng = np.random.RandomState(random_state)
        self.estimators = [model_class(*args, random_state=self.rng, **kwargs) for _ in range(self.n_member)]

    def fit(self, X, y, majority=1):
        one_indices = np.where(y == 1)[0]
        zero_indices = np.where(y == 0)[0]

        # Downsample majority
        for i in range(self.n_member):
            indices = self.rng.choice([zero_indices, one_indices][majority], size=len([zero_indices, one_indices][1-majority]), replace=False)
            indices = np.concatenate([indices, [zero_indices, one_indices][1-majority]])
            _x = X[indices]
            _y = y[indices]
            self.estimators[i].fit(_x, _y)

    def predict(self, X, thresh=0.5):
        preds = np.vstack([(self.estimators[i].predict_proba(X)[:, 1] >= thresh).astype(np.int8) for i in range(self.n_member)])
        return mode(preds, keepdims=True)[0].squeeze()

class UpsampleEnsembleClassifier:

    def __init__(self, model_class, n_member, *args, random_state=None, **kwargs):
        self.n_member = n_member
        self.rng = np.random.RandomState(random_state)
        self.estimators = [model_class(*args, random_state=self.rng, **kwargs) for _ in range(self.n_member)]

    def fit(self, X, y, minority=0):
        one_indices = np.where(y == 1)[0]
        zero_indices = np.where(y == 0)[0]

        # Upsample minority
        for i in range(self.n_member):
            # Upsample minority
            indices = self.rng.choice([zero_indices, one_indices][minority], size=len([zero_indices, one_indices][1-minority]), replace=True)
            indices = np.concatenate([indices, [zero_indices, one_indices][1-minority]])
            _x = X[indices]
            _y = y[indices]
            self.estimators[i].fit(_x, _y)

    def predict(self, X, thresh=0.5):
        preds = np.vstack([(self.estimators[i].predict_proba(X)[:, 1] >= thresh).astype(np.int8) for i in range(self.n_member)])
        return mode(preds, keepdims=True)[0].squeeze()

# Scikit-learn ensemble with median prediction
class Ensemble:

    def __init__(self, base_estimator, N, *args, **kwargs):
        self.estimators = [base_estimator(*args, **kwargs) for _ in range(N)]

    def fit(self, *args, **kwargs):
        for estimator in self.estimators:
            estimator.fit(*args, **kwargs)

    def predict(self, X):
        preds = []
        for estimator in self.estimators:
            preds.append(estimator.predict(X).reshape(-1, 1))
        
        return np.median(np.concatenate(preds, axis=-1), axis=-1).squeeze()

# Parse sklearn ensemble into pytorch ensemble to explain with captum
class PyTorchEnsemble(nn.Module):

    def __init__(self, scikit_ensemble):
        super().__init__()
        self.scikit_ensemble = scikit_ensemble
        self.ensemble = []

        for estimator in scikit_ensemble.estimators:
            n_layers = len(estimator.coefs_)
            model = nn.Sequential()
            for i in range(n_layers):
                n_in, n_out = estimator.coefs_[i].shape
                layer = nn.Linear(n_in, n_out)
                layer.weight = torch.nn.Parameter(torch.from_numpy(estimator.coefs_[i].T))
                layer.bias = torch.nn.Parameter(torch.from_numpy(estimator.intercepts_[i]))
                model.append(layer)
                # Add ReLU if not output layer
                if i != n_layers-1:
                    model.append(nn.ReLU())

            self.ensemble.append(model)

    def predict(self, X):
        return self.scikit_ensemble.predict(X)

    def forward(self, X, y):
        preds = []
        for estimator in self.ensemble:
            preds.append(estimator(X).reshape(-1, 1))
        
        return (torch.median(torch.cat(preds, axis=-1), axis=-1).values.squeeze() - y)**2


class PyTorchLinear(nn.Module):

    def __init__(self, scikit_linear):
        super().__init__()
        self.scikit_linear = scikit_linear
        n_features = self.scikit_linear.coef_.shape[-1]
        self.lin = nn.Linear(n_features, 1)
        self.lin.weight = torch.nn.Parameter(torch.from_numpy(scikit_linear.coef_))
        self.lin.bias = torch.nn.Parameter(torch.from_numpy(scikit_linear.intercept_))

    def predict(self, X):
        return self.scikit_linear.predict(X)

    def forward(self, X, y):
        return (self.lin(X).squeeze() - y)**2