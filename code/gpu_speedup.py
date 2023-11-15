import torch
import torch.nn as nn
import time
import numpy as np
import pickle
import tqdm

from models import Ensemble, PyTorchEnsemble, PyTorchLinear
from captum.attr import DeepLiftShap
from tsx.datasets.monash import load_monash
from tsx.datasets import windowing

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def get_explanations(model, X, y, background):
    explainer = DeepLiftShap(model)
    explanations = []

    for i in tqdm.trange(len(X)):
        explanations.append(explainer.attribute(X[i].reshape(1, -1), background, additional_forward_args=y[i].reshape(1, -1)).detach().cpu().numpy().reshape(1, -1))

    return np.concatenate(explanations, axis=0)

def main():

    L = 10
    d_idx = 0

    _X, horizons = load_monash('weather', return_horizon=True)
    for d_idx in range(10):
        X = _X['series_value'][d_idx]
        H = horizons[d_idx]

        # Split and normalize data
        end_train = int(len(X) * 0.5)
        end_val = end_train + int(len(X) * 0.25)
        X_train = X[:end_train]
        X_val = X[end_train:end_val]

        mu = np.mean(X_train)
        std = np.std(X_train)

        X = (X - mu) / std

        X_train = X[:end_train]
        X_val = X[end_train:end_val]

        # Instead of forecasting t+1, forecast t+j
        x_train, y_train = windowing(X_train, L=L, H=H)
        x_val, y_val = windowing(X_val, L=L, H=H)
        y_train = y_train[..., -1:]
        y_val = y_val[..., -1:]

        x_train = x_train.astype(np.float32)
        x_val = x_val.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)

        X = torch.from_numpy(x_val).float()
        y = torch.from_numpy(y_val).float()
        background = torch.from_numpy(x_train).float()[:1000]

        with open(f'models/weather/{d_idx}/nn.pickle', 'rb') as f:
            old_ensemble = pickle.load(f)

        ensemble = PyTorchEnsemble(old_ensemble)

        print('-'*80)
        print(d_idx)
        print('-'*80)
        print('x - y - bg', X.shape, y.shape, background.shape)

        # ---------- No GPU
        before = time.time()
        explanations = get_explanations(ensemble, X, y, background)
        after = time.time()
        print('no gpu', after-before)

        # ---------- With GPU
        before = time.time()
        X = X.to('cuda')
        y = y.to('cuda')
        background = background.to('cuda')
        for estimator in ensemble.ensemble:
            estimator.to('cuda')
        explanations = get_explanations(ensemble, X, y, background)
        after = time.time()
        print('with gpu', after-before)
        print('-'*80)
        print(' ')

if __name__ == '__main__':
    main()
