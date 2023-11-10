import numpy as np
import torch
import torch.nn as nn
import pickle

from main import Ensemble
from captum.attr import DeepLiftShap
from sklearn.linear_model import LinearRegression

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

def get_explanations(model, X, y, background):
    explainer = DeepLiftShap(model)
    explanations = explainer.attribute(X, background, additional_forward_args=(y,)).detach().cpu().numpy()

    return explanations

def main():
    with open('models/weather/0/nn.pickle', 'rb') as f:
        old_ensemble = pickle.load(f)

    ensemble = PyTorchEnsemble(old_ensemble)
    X = torch.rand((16, 10))
    y = torch.rand((16))
    background = torch.rand((64, 10))

    explanations = get_explanations(ensemble, X, y, background)
    print('nn', explanations.shape)

    # ----------------------------------
    with open('models/weather/0/linear.pickle', 'rb') as f:
        lin = pickle.load(f)

    model = PyTorchLinear(lin)
    explanations = get_explanations(model, X, y, background)
    print('lin', explanations.shape)

if __name__ == '__main__':
    main()