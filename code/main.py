import tqdm
import skorch
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time

from tsx.datasets.monash import load_m4_daily_bench, load_monash
from tsx.datasets import windowing
from tsx.models.forecaster import NLinear
from tsx.models.forecaster.model_zoo import get_1d_cnn
from tsx.models import NeuralNetRegressor
from seedpy import fixedseed
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from seedpy import fixedseed
from os import makedirs
from joblib import Parallel, delayed
from os.path import exists

from cdd_plots import create_cdd
from models import PyTorchLinear, PyTorchEnsemble
from selection import run_v1, run_v2, run_v3, run_v4, run_v5, run_v6, selection_oracle_percent, run_v7, run_test, run_v8, run_v9, run_v10, run_sebas_selection, run_v11
from viz import plot_rocs
from datasets import load_dataset
from explainability import get_explanations
from tsx.model_selection import ROC_Member
from tsx.models.forecaster import split_zero

class MedianPredictionEnsemble:

    def __init__(self, estimators):
        self.estimators = estimators

    @classmethod
    def load_model(cls, ds_name, ds_index, n=10):
        estimators = []
        for i in range(n):
            with open(f'models/{ds_name}/{ds_index}/nns/{i}.pickle', 'rb') as f:
                estimators.append(pickle.load(f))

        return cls(estimators)

    def save_model(self, ds_name, ds_index):
        makedirs(f'models/{ds_name}/{ds_index}/nns/', exist_ok=True)
        for idx, estimator in enumerate(self.estimators):
            with open(f'models/{ds_name}/{ds_index}/nns/{idx}.pickle', 'wb+') as f:
                pickle.dump(estimator, f)


    def fit(self, *args, **kwargs):
        for estimator in self.estimators:
            estimator.fit(*args, **kwargs)

    def predict(self, X):
        preds = []
        for estimator in self.estimators:
            preds.append(estimator.predict(X).reshape(-1, 1))
        
        return np.median(np.concatenate(preds, axis=-1), axis=-1).squeeze()


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

def main(to_run):
    L = 10

    ds_names = ['kdd_cup_nomissing', 'pedestrian_counts', 'weather', 'web_traffic']

    if to_run is None:
        to_run = []

    for ds_name in ds_names:
        log_test = []
        log_selection = []

        X, horizons, indices = load_dataset(ds_name, fraction=1)

        if ds_name == 'web_traffic':
            lr = 1e-5
            max_epochs = 10000
        else:
            lr = 1e-3
            max_epochs = 500

        print(ds_name, 'n_series', len(indices))

        # for i in [3401]:
        #     print('-'*30, i, '-'*30)
        #     log_test, log_selection = run_experiment(ds_name, i, X[i], L, horizons[i], max_iter_nn=max_epochs)
        # exit()

        # Load previous results (if available), otherwise create empty
        if exists(f'results/{ds_name}_test.csv') and exists(f'results/{ds_name}_selection.csv'):
            test_results = pd.read_csv(f'results/{ds_name}_test.csv')
            test_results = test_results.set_index('dataset_names')
            test_results = test_results.T.to_dict()
            test_selection = pd.read_csv(f'results/{ds_name}_selection.csv')
            test_selection = test_selection.set_index('dataset_names')
            test_selection = test_selection.T.to_dict()
        else:
            test_results = [{} for _ in range(len(X))]
            test_selection = [{} for _ in range(len(X))]

        print('To run', to_run)

        log_test, log_selection = zip(*Parallel(n_jobs=-1, backend='loky')(delayed(run_experiment)(ds_name, ds_index, X[ds_index], L, horizons[ds_index], test_results[ds_index], test_selection[ds_index], lr=lr, max_iter_nn=max_epochs, to_run=to_run) for ds_index in tqdm.tqdm(indices)))
        # ds_index = 0
        # run_experiment(ds_name, ds_index, X[ds_index], L, horizons[ds_index], test_results[ds_index], test_selection[ds_index], lr=lr, max_iter_nn=max_epochs, to_run=to_run)
        log_test = pd.DataFrame(list(log_test))
        log_test = log_test.set_index('dataset_names')
        log_test.to_csv(f'results/{ds_name}_test.csv')
        log_selection = pd.DataFrame(list(log_selection))
        log_selection = log_selection.set_index('dataset_names')
        log_selection.to_csv(f'results/{ds_name}_selection.csv')

def run_experiment(ds_name, ds_index, X, L, H, test_results, selection_results, lr=1e-3, max_iter_nn=500, to_run=None):
    #print(ds_index)
    makedirs(f'models/{ds_name}/{ds_index}', exist_ok=True)

    test_results['dataset_names'] = ds_index
    selection_results['dataset_names'] = ds_index

    #print('rerun', to_run)

    # Split and normalize data
    end_train = int(len(X) * 0.5)
    end_val = end_train + int(len(X) * 0.25)
    X_train = X[:end_train]
    X_val = X[end_train:end_val]
    X_test = X[end_val:]

    mu = np.mean(X_train)
    std = np.std(X_train)

    X = (X - mu) / std

    X_train = X[:end_train]
    X_val = X[end_train:end_val]
    X_test = X[end_val:]

    # Instead of forecasting t+1, forecast t+j
    x_train, y_train = windowing(X_train, L=L, H=H)
    x_val, y_val = windowing(X_val, L=L, H=H)
    x_test, y_test = windowing(X_test, L=L, H=H)
    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)
    if len(y_val.shape) == 1:
        y_val = y_val.reshape(-1, 1)
    if len(y_test.shape) == 1:
        y_test = y_test.reshape(-1, 1)
    y_train = y_train[..., -1:]
    y_val = y_val[..., -1:]
    y_test = y_test[..., -1:]

    x_train = x_train.astype(np.float32)
    x_val = x_val.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # Train base models
    try:
        with open(f'models/{ds_name}/{ds_index}/linear.pickle', 'rb') as f:
            f_i = pickle.load(f)
    except Exception:
        f_i = LinearRegression()
        f_i.fit(x_train, y_train)
        with open(f'models/{ds_name}/{ds_index}/linear.pickle', 'wb') as f:
            pickle.dump(f_i, f)

    # Neural net
    try:
        f_c = MedianPredictionEnsemble.load_model(ds_name, ds_index)
    except Exception:
        #print(f'retrain nns for {ds_name}/{ds_index}')
        with fixedseed(np, 20231103):
            f_c = MedianPredictionEnsemble([MLPRegressor((28,), learning_rate_init=lr, max_iter=max_iter_nn) for _ in range(10)])
            f_c.fit(x_train, y_train.squeeze())
            f_c.save_model(ds_name, ds_index)

    makedirs(f'preds/{ds_name}/{ds_index}', exist_ok=True)

    lin_preds_train = f_i.predict(x_train)
    lin_preds_val = f_i.predict(x_val)
    lin_preds_test = f_i.predict(x_test)
    loss_i_test = rmse(lin_preds_test, y_test)
    np.save(f'preds/{ds_name}/{ds_index}/lin.npy', lin_preds_test.reshape(-1))
    test_results['linear'] = loss_i_test

    nn_preds_train = f_c.predict(x_train)
    nn_preds_val = f_c.predict(x_val)
    nn_preds_test = f_c.predict(x_test)
    loss_nn_test = rmse(nn_preds_test, y_test)
    np.save(f'preds/{ds_name}/{ds_index}/nn.npy', nn_preds_test.reshape(-1))
    test_results['nn'] = loss_nn_test

    closeness_threshold = 1e-6

    lin_val_error = (lin_preds_val.squeeze()-y_val.squeeze())**2
    nn_val_error = (nn_preds_val.squeeze()-y_val.squeeze())**2
    lin_val_better = np.where(lin_val_error-nn_val_error <= closeness_threshold)[0]
    nn_val_better = np.array([idx for idx in range(len(lin_val_error)) if idx not in lin_val_better]).astype(np.int32)
    assert len(lin_val_better) + len(nn_val_better) == len(lin_val_error)

    lin_test_error = (lin_preds_test.squeeze()-y_test.squeeze())**2
    nn_test_error = (nn_preds_test.squeeze()-y_test.squeeze())**2
    lin_test_better = np.where(lin_test_error-nn_test_error <= closeness_threshold)[0]
    nn_test_better = np.array([idx for idx in range(len(lin_test_error)) if idx not in lin_test_better]).astype(np.int32)
    assert len(lin_test_better) + len(nn_test_better) == len(lin_test_error)

    # -------------------------------- Model selection

    lin_preds_val = lin_preds_val.squeeze()
    lin_preds_test = lin_preds_test.squeeze()
    nn_preds_val = nn_preds_val.squeeze()
    nn_preds_test = nn_preds_test.squeeze()
    y_val = y_val.squeeze()
    y_test = y_test.squeeze()

    # TODO: Rewrite with closeness threshold
    #optimal_selection_val = (lin_preds_val-y_val)**2 <= (nn_preds_val-y_val)**2
    #optimal_selection_test = (lin_preds_test-y_test)**2 <= (nn_preds_test-y_test)**2
    optimal_selection_test = (lin_test_error - nn_test_error) <= closeness_threshold

    optimal_prediction_test = np.choose(optimal_selection_test, [nn_preds_test, lin_preds_test])
    loss_optimal_test = rmse(optimal_prediction_test, y_test)
    test_results['selOpt'] = loss_optimal_test
    selection_results['selOpt'] = np.mean(optimal_selection_test)

    # -------------------------------- Binomial baseline

    # rng = np.random.RandomState(934878167)
    # repeats = 3
    
    # for p in [0.9, 0.95, 0.99]:
    #     loss_binom_val = 0
    #     loss_binom_test = 0
    #     for _ in range(repeats):
    #         binom_selection_val = rng.binomial(1, p, size=len(x_val))
    #         binom_selection_test = rng.binomial(1, p, size=len(x_test))

    #         binom_prediction_val = np.choose(binom_selection_val, [nn_preds_val, lin_preds_val])
    #         binom_prediction_test = np.choose(binom_selection_test, [nn_preds_test, lin_preds_test])
    #         loss_binom_val += rmse(binom_prediction_val, y_val)
    #         loss_binom_test += rmse(binom_prediction_test, y_test)

    #     loss_binom_val /= repeats
    #     loss_binom_test /= repeats

    #     test_results[f'selBinom{p}'] = loss_binom_test
    #     selection_results[f'selBinom{p}'] = np.mean(binom_selection_test)

    #return test_results, selection_results

    # -------------------------------- RoC based methods

    # Compute explanations on validation data points
    # try:
    #     lin_expl = np.load(f'explanations/{ds_name}/{ds_index}/lin_expl.npy')
    #     ensemble_expl = np.load(f'explanations/{ds_name}/{ds_index}/ensemble_expl.npy')
    # except Exception:
    #     rng = np.random.RandomState(20231110 + ds_index)
    #     indices = rng.choice(np.arange(len(x_train)), size=min(1000, len(x_train)), replace=False)
    #     background = x_train[indices]

    #     # If available, use gpu acceleration
    #     pt_xval_lin = torch.from_numpy(x_val[lin_better])
    #     pt_yval_lin = torch.from_numpy(y_val[lin_better])

    #     pt_xval_ens = torch.from_numpy(x_val[nn_better])
    #     pt_yval_ens = torch.from_numpy(y_val[nn_better])

    #     pt_background = torch.from_numpy(background)

    #     with fixedseed([torch, np], seed=(20231110+ds_index)):
    #         pt_linear = PyTorchLinear(f_i)
    #         pt_ensemble = PyTorchEnsemble(f_c)
    #         lin_expl = get_explanations(pt_linear, pt_xval_lin, pt_yval_lin, pt_background)
    #         ensemble_expl = get_explanations(pt_ensemble, pt_xval_ens, pt_yval_ens, pt_background)
    #     makedirs(f'explanations/{ds_name}/{ds_index}', exist_ok=True)
    #     np.save(f'explanations/{ds_name}/{ds_index}/lin_expl.npy', lin_expl)
    #     np.save(f'explanations/{ds_name}/{ds_index}/ensemble_expl.npy', ensemble_expl)

    # # Load (or compute) rocs
    # if x_val[lin_better].shape != lin_expl.shape:
    #     if x_val[lin_better].shape[0] == 0:
    #         lin_rocs = []
    #     else:
    #         print('redo', ds_index, x_val[lin_better].shape, lin_expl.shape)
    #         return test_results, selection_results
    # else:
    #     lin_rocs = compute_rocs(x_val[lin_better], y_val[lin_better], lin_expl, lin_error[lin_better])

    # if x_val[nn_better].shape != ensemble_expl.shape:
    #     if x_val[nn_better].shape[0] == 0:
    #         ensemble_rocs = []
    #     else:
    #         print('redo', ds_index, x_val[nn_better].shape, ensemble_expl.shape)
    #         return test_results, selection_results
    # else:
    #     ensemble_rocs = compute_rocs(x_val[nn_better], y_val[nn_better], ensemble_expl, nn_error[nn_better])

    # ------------------ Run new selection methods

    # Save train data for later
    # from selection import get_roc_dists
    # if len(lin_rocs) > 0 and len(ensemble_rocs) > 0:
    #     lin_min_dist, ensemble_min_dist = get_roc_dists(x_val, lin_rocs, ensemble_rocs)
    #     save_y = y_val
    #     save_x = np.vstack([lin_preds_val, nn_preds_val, lin_min_dist, ensemble_min_dist]).T

    #     makedirs(f'data/optim_{ds_name}/{ds_index}', exist_ok=True)
    #     np.save(f'data/optim_{ds_name}/{ds_index}/train_X.npy', save_x)
    #     np.save(f'data/optim_{ds_name}/{ds_index}/train_y.npy', save_y)
    #     lin_min_dist, ensemble_min_dist = get_roc_dists(x_test, lin_rocs, ensemble_rocs)
    #     np.save(f'data/optim_{ds_name}/{ds_index}/test_X.npy', np.vstack([lin_preds_test, nn_preds_test, lin_min_dist, ensemble_min_dist]).T)
    #     np.save(f'data/optim_{ds_name}/{ds_index}/test_y.npy', y_test)

    #return test_results, selection_results
    # v4
    # if 'v4' in to_run or 'v4_0.5' not in test_results.keys():
    #     name, test_selection = run_v4(optimal_selection_val, x_val, x_test, lin_preds_val, nn_preds_val, lin_preds_test, nn_preds_test, lin_rocs, ensemble_rocs, random_state=20231116+ds_index)
    #     test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
    #     loss_test_test = rmse(test_prediction_test, y_test)
    #     np.save(f'preds/{ds_name}/{ds_index}/v4.npy', test_prediction_test.reshape(-1))
    #     test_results[name] = loss_test_test
    #     selection_results[name] = np.mean(test_selection)

        # name, test_selection = run_v4(optimal_selection_val, x_val, x_test, lin_preds_val, nn_preds_val, lin_preds_test, nn_preds_test, lin_rocs, ensemble_rocs, random_state=20231116+ds_index, calibrate=True)
        # test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
        # loss_test_test = rmse(test_prediction_test, y_test)
        # test_results[name] = loss_test_test
        # selection_results[name] = np.mean(test_selection)

    # v5
    # if 'v5' in to_run or 'v5' not in test_results.keys():
    #     name, test_selection = run_v5(x_val, y_val, x_test, lin_preds_val, nn_preds_val, lin_preds_test, nn_preds_test, lin_rocs, ensemble_rocs, random_state=20231222+ds_index)
    #     test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
    #     loss_test_test = rmse(test_prediction_test, y_test)
    #     np.save(f'preds/{ds_name}/{ds_index}/v5.npy', test_prediction_test.reshape(-1))
    #     test_results[name] = loss_test_test
    #     selection_results[name] = np.mean(test_selection)

    # if 'v8' in to_run or 'v8' not in test_results.keys():
    #     name, test_selection = run_v8(x_val, y_val, x_test, lin_preds_val, nn_preds_val, lin_preds_test, nn_preds_test, lin_rocs, ensemble_rocs, random_state=20231222+ds_index)
    #     test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
    #     loss_test_test = rmse(test_prediction_test, y_test)
    #     np.save(f'preds/{ds_name}/{ds_index}/v8.npy', test_prediction_test.reshape(-1))
    #     test_results[name] = loss_test_test
    #     selection_results[name] = np.mean(test_selection)

    # if 'v9' in to_run or 'v9' not in test_results.keys():
    #     name, test_selection = run_v9(x_val, y_val, x_test, lin_preds_val, nn_preds_val, lin_preds_test, nn_preds_test, lin_rocs, ensemble_rocs, random_state=20231222+ds_index)
    #     test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
    #     loss_test_test = rmse(test_prediction_test, y_test)
    #     np.save(f'preds/{ds_name}/{ds_index}/v9.npy', test_prediction_test.reshape(-1))
    #     test_results[name] = loss_test_test
    #     selection_results[name] = np.mean(test_selection)

    # if 'v10' in to_run or 'v10' not in test_results.keys():
    #     name, test_selection = run_v10(y_train, x_val, y_val, x_test, y_test, lin_preds_train, nn_preds_train, lin_preds_val, nn_preds_val, lin_preds_test, nn_preds_test, lin_rocs, ensemble_rocs, random_state=20231322+ds_index)
    #     test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
    #     loss_test_test = rmse(test_prediction_test, y_test)
    #     np.save(f'preds/{ds_name}/{ds_index}/v10.npy', test_prediction_test.reshape(-1))
    #     test_results[name] = loss_test_test
    #     selection_results[name] = np.mean(test_selection)

    if 'v11' in to_run or 'v11' not in test_results.keys():
        name, test_selection = run_v11(x_val, y_val, x_test, lin_preds_val, nn_preds_val, lin_preds_test, nn_preds_test, random_state=20231322+ds_index, p=0.9)
        test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
        loss_test_test = rmse(test_prediction_test, y_test)
        np.save(f'preds/{ds_name}/{ds_index}/{name}.npy', test_prediction_test.reshape(-1))
        test_results[name] = loss_test_test
        selection_results[name] = np.mean(test_selection)

        name, test_selection = run_v11(x_val, y_val, x_test, lin_preds_val, nn_preds_val, lin_preds_test, nn_preds_test, random_state=20231322+ds_index, p=0.8)
        test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
        loss_test_test = rmse(test_prediction_test, y_test)
        np.save(f'preds/{ds_name}/{ds_index}/{name}.npy', test_prediction_test.reshape(-1))
        test_results[name] = loss_test_test
        selection_results[name] = np.mean(test_selection)

        name, test_selection = run_v11(x_val, y_val, x_test, lin_preds_val, nn_preds_val, lin_preds_test, nn_preds_test, random_state=20231322+ds_index, p=0.7)
        test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
        loss_test_test = rmse(test_prediction_test, y_test)
        np.save(f'preds/{ds_name}/{ds_index}/{name}.npy', test_prediction_test.reshape(-1))
        test_results[name] = loss_test_test
        selection_results[name] = np.mean(test_selection)

    # if 'test' in to_run:
    #     name, test_selection =  run_test(y_val, y_test, lin_preds_val, nn_preds_val, lin_preds_test, nn_preds_test, epsilon=1)
    #     test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
    #     loss_test_test = rmse(test_prediction_test, y_test)
    #     name = name + '_1.0'
    #     np.save(f'preds/{ds_name}/{ds_index}/{name}.npy', test_prediction_test.reshape(-1))
    #     test_results[name] = loss_test_test
    #     selection_results[name] = np.mean(test_selection)

    #     name, test_selection =  run_test(y_val, y_test, lin_preds_val, nn_preds_val, lin_preds_test, nn_preds_test, epsilon=1.1)
    #     test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
    #     loss_test_test = rmse(test_prediction_test, y_test)
    #     name = name + '_1.1'
    #     np.save(f'preds/{ds_name}/{ds_index}/{name}.npy', test_prediction_test.reshape(-1))
    #     test_results[name] = loss_test_test
    #     selection_results[name] = np.mean(test_selection)

    # ------------------ Baseline to check if v4 is actually better
    # name = 'selBinomOptP'
    # p_optimal_sel = np.mean(optimal_selection_val)
    # sel_rng = np.random.RandomState(20231208)
    # test_prediction_test = np.zeros((len(y_test)))
    # n_runs = 10
    # loss_test_test = 0
    # for _ in range(n_runs):
    #     test_selection = sel_rng.binomial(1, p_optimal_sel, size=len(y_test))
    #     test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
    #     loss_test_test += rmse(test_prediction_test, y_test)

    # loss_test_test /= n_runs
    # test_results[name] = loss_test_test

    # Selection oracle
    name = 'ErrorOracle90'
    test_selection = selection_oracle_percent(y_test, lin_preds_test, nn_preds_test, 0.9)
    test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
    loss_test_test = rmse(test_prediction_test, y_test)
    np.save(f'preds/{ds_name}/{ds_index}/{name}.npy', test_prediction_test.reshape(-1))
    test_results[name] = loss_test_test
    selection_results[name] = np.mean(test_selection)

    # name, test_selection = run_sebas_selection(y_test, lin_preds_test, nn_preds_test, 3, 3)
    # name = name + "3,3"
    # test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
    # loss_test_test = rmse(test_prediction_test, y_test)
    # np.save(f'preds/{ds_name}/{ds_index}/{name}.npy', test_prediction_test.reshape(-1))
    # test_results[name] = loss_test_test
    # selection_results[name] = np.mean(test_selection)

    # name, test_selection = run_sebas_selection(y_test, lin_preds_test, nn_preds_test, 8, 2)
    # name = name + "8,2"
    # test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
    # loss_test_test = rmse(test_prediction_test, y_test)
    # np.save(f'preds/{ds_name}/{ds_index}/{name}.npy', test_prediction_test.reshape(-1))
    # test_results[name] = loss_test_test
    # selection_results[name] = np.mean(test_selection)

    return test_results, selection_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--override", help='', nargs='+', default=[])
    args = vars(parser.parse_args())
    main(args['override'])
