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

from cdd_plots import create_cdd
from models import PyTorchLinear, PyTorchEnsemble
from selection import run_v1, run_v2, run_v3, run_v4, run_v5, run_v6, selection_oracle_percent
from viz import plot_rocs
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

def main():
    L = 10

    '''
    ### Australian electricity demands
    X, horizons = load_monash('australian_electricity_demand', return_horizon=True)
    X = X['series_value']
    H = horizons[0]
    log_val = []
    log_test = []
    for i, x in enumerate(X):
        print('-'*30, i, '-'*30)
        log_val, log_test = run_experiment(log_val, log_test, 'australian_electricity_demand', i, x, L, H, verbose=True)

    log_val = pd.DataFrame(log_val)
    log_val.index.rename('dataset_names', inplace=True)
    log_val.to_csv('results/australian_electricity_demand_val.csv')
    log_test = pd.DataFrame(log_test)
    log_test.index.rename('dataset_names', inplace=True)
    log_test.to_csv('results/australian_electricity_demand_test.csv')

    create_cdd('australian_electricity_demand')

    ### KDD Cup
    X, horizons = load_monash('kdd_cup_nomissing', return_horizon=True)
    X = X['series_value']
    log_val = []
    log_test = []
    for i, x in enumerate(X):
        print('-'*30, i, '-'*30)
        log_val, log_test = run_experiment(log_val, log_test, 'kdd', i, x, L, horizons[i], verbose=True)

    log_val = pd.DataFrame(log_val)
    log_val.index.rename('dataset_names', inplace=True)
    log_val.to_csv('results/kdd_val.csv')
    log_test = pd.DataFrame(log_test)
    log_test.index.rename('dataset_names', inplace=True)
    log_test.to_csv('results/kdd_test.csv')

    create_cdd('kdd')
    '''

    ### weather
    X, horizons = load_monash('weather', return_horizon=True)
    X = X['series_value']
    log_val = []
    log_test = []
    log_selection = []

    # Choose subset
    rng = np.random.RandomState(12389182)
    run_size = len(X)
    #run_size = 600
    indices = rng.choice(np.arange(len(X)), size=run_size, replace=False)

    # indices = rng.choice(np.arange(len(X)), size=10, replace=False)
    # for i in indices:
    #     print('-'*30, i, '-'*30)
    #     log_val, log_test, log_selection = run_experiment('weather', i, X[i], L, horizons[i])
    # exit()

    log_val, log_test, log_selection = zip(*Parallel(n_jobs=-1, backend="loky")(delayed(run_experiment)('weather', ds_index, X[ds_index], L, horizons[ds_index]) for ds_index in tqdm.tqdm(indices)))

    log_val = pd.DataFrame(list(log_val))
    log_val.index.rename('dataset_names', inplace=True)
    log_val.to_csv('results/weather_val.csv')
    log_test = pd.DataFrame(list(log_test))
    log_test.index.rename('dataset_names', inplace=True)
    log_test.to_csv('results/weather_test.csv')
    log_selection = pd.DataFrame(list(log_selection))
    log_selection.index.rename('dataset_names', inplace=True)
    log_selection.to_csv('results/weather_selection.csv')

    create_cdd('weather')

def run_experiment(ds_name, ds_index, X, L, H, lr=1e-3):
    #print(ds_index)
    makedirs(f'models/{ds_name}/{ds_index}', exist_ok=True)

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
    y_train = y_train[..., -1:]
    y_val = y_val[..., -1:]
    y_test = y_test[..., -1:]

    x_train = x_train.astype(np.float32)
    x_val = x_val.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    y_test = y_test.astype(np.float32)

    val_results = {}
    test_results = {}
    selection_results = {}

    # Train base models
    try:
        with open(f'models/{ds_name}/{ds_index}/linear.pickle', 'rb') as f:
            f_i = pickle.load(f)
    except Exception:
        f_i = LinearRegression()
        f_i.fit(x_train, y_train)
        with open(f'models/{ds_name}/{ds_index}/linear.pickle', 'wb') as f:
            pickle.dump(f_i, f)

    lin_preds_val = f_i.predict(x_val)
    lin_preds_test = f_i.predict(x_test)
    loss_i_val = rmse(lin_preds_val, y_val)
    loss_i_test = rmse(lin_preds_test, y_test)
    val_results['linear'] = loss_i_val
    test_results['linear'] = loss_i_test

    # Neural net
    try:
        # with open(f'models/{ds_name}/{ds_index}/nn.pickle', 'rb') as f:
        #     f_c = pickle.load(f)
        f_c = MedianPredictionEnsemble.load_model(ds_name, ds_index)
    except Exception:
        print(f'retrain nns for {ds_name}/{ds_index}')
        with fixedseed(np, 20231103):
            f_c = MedianPredictionEnsemble([MLPRegressor((28,), learning_rate_init=lr, max_iter=500) for _ in range(10)])
            f_c.fit(x_train, y_train.squeeze())
            f_c.save_model(ds_name, ds_index)
        # with open(f'models/{ds_name}/{ds_index}/nn.pickle', 'wb') as f:
        #     pickle.dump(f_c, f)

    nn_preds_val = f_c.predict(x_val)
    nn_preds_test = f_c.predict(x_test)
    loss_nn_val = rmse(nn_preds_val, y_val)
    loss_nn_test = rmse(nn_preds_test, y_test)
    val_results['nn'] = loss_nn_val
    test_results['nn'] = loss_nn_test

    # Baselines
    loss_lv_val = rmse(x_val[..., -1:], y_val)
    loss_lv_test = rmse(x_test[..., -1:], y_test)
    loss_mean_val = rmse(x_val.mean(axis=-1).reshape(-1, 1), y_val) 
    loss_mean_test = rmse(x_test.mean(axis=-1).reshape(-1, 1), y_test) 
    # val_results['lv'] = loss_lv_val
    # test_results['lv'] = loss_lv_test
    # val_results['mean'] = loss_mean_val
    # test_results['mean'] = loss_mean_test

    #####################################
    # Model selection
    #####################################
    lin_preds_val = lin_preds_val.squeeze()
    lin_preds_test = lin_preds_test.squeeze()
    nn_preds_val = nn_preds_val.squeeze()
    nn_preds_test = nn_preds_test.squeeze()
    y_val = y_val.squeeze()
    y_test = y_test.squeeze()

    optimal_selection_val = (lin_preds_val-y_val)**2 <= (nn_preds_val-y_val)**2
    optimal_selection_test = (lin_preds_test-y_test)**2 <= (nn_preds_test-y_test)**2

    optimal_prediction_val = np.choose(optimal_selection_val, [nn_preds_val, lin_preds_val])
    optimal_prediction_test = np.choose(optimal_selection_test, [nn_preds_test, lin_preds_test])
    loss_optimal_val = rmse(optimal_prediction_val, y_val)
    loss_optimal_test = rmse(optimal_prediction_test, y_test)
    val_results['selOpt'] = loss_optimal_val
    test_results['selOpt'] = loss_optimal_test
    selection_results['selOpt'] = np.mean(optimal_selection_test)

    for p_lin in [0.9, 0.95, 0.99]:
        oracle_selection = selection_oracle_percent(y_test, lin_preds_test, nn_preds_test, p_lin)
        oracle_prediction_test = np.choose(oracle_selection, [nn_preds_test, lin_preds_test])
        loss_oracle_test = rmse(oracle_prediction_test, y_test)
        test_results[f'oracle_{p_lin}'] = loss_oracle_test
        selection_results[f'oracle_{p_lin}'] = np.mean(oracle_selection)

    # -------------------------------- Threshold baseline
    for thresh in [0.5, 0.99, 1.5, 1.7]:
        thresh_selection_val = (lin_preds_val-nn_preds_val)**2 / (lin_preds_val**2 + nn_preds_val**2) <= thresh
        thresh_selection_test = (lin_preds_test-nn_preds_test)**2 / (lin_preds_test**2 + nn_preds_test**2) <= thresh
        # thresh_selection_val = (lin_preds_val-nn_preds_val)**2 <= thresh
        # thresh_selection_test = (lin_preds_test-nn_preds_test)**2 <= thresh

        thresh_prediction_val = np.choose(thresh_selection_val, [nn_preds_val, lin_preds_val])
        thresh_prediction_test = np.choose(thresh_selection_test, [nn_preds_test, lin_preds_test])
        loss_thresh_val = rmse(thresh_prediction_val, y_val)
        loss_thresh_test = rmse(thresh_prediction_test, y_test)
        # val_results[f'selThresh{thresh}'] = loss_thresh_val
        # test_results[f'selThresh{thresh}'] = loss_thresh_test

    # -------------------------------- Binomial baseline
    rng = np.random.RandomState(934878167)
    repeats = 3
    
    for p in [0.9, 0.95, 0.99]:
        loss_binom_val = 0
        loss_binom_test = 0
        for _ in range(repeats):
            binom_selection_val = rng.binomial(1, p, size=len(x_val))
            binom_selection_test = rng.binomial(1, p, size=len(x_test))

            binom_prediction_val = np.choose(binom_selection_val, [nn_preds_val, lin_preds_val])
            binom_prediction_test = np.choose(binom_selection_test, [nn_preds_test, lin_preds_test])
            loss_binom_val += rmse(binom_prediction_val, y_val)
            loss_binom_test += rmse(binom_prediction_test, y_test)

        loss_binom_val /= repeats
        loss_binom_test /= repeats

        val_results[f'selBinom{p}'] = loss_binom_val
        test_results[f'selBinom{p}'] = loss_binom_test
        selection_results[f'selBinom{p}'] = np.mean(binom_selection_test)

    # -------------------------------- RoC based methods

    makedirs(f'explanations/{ds_name}/{ds_index}', exist_ok=True)
    pt_linear = PyTorchLinear(f_i)
    pt_ensemble = PyTorchEnsemble(f_c)

    # Find best models on each validation datapoint
    lin_error = (pt_linear.predict(x_val).squeeze() - y_val)**2
    ensemble_error = (pt_ensemble.predict(x_val).squeeze() - y_val)**2
    selection = (lin_error <= ensemble_error)
    lin_indices = np.where(selection)[0]
    ensemble_indices = np.where(~selection)[0]
    
    # Compute explanations on validation data points
    try:
        lin_expl = np.load(f'explanations/{ds_name}/{ds_index}/lin_expl.npy')
        ensemble_expl = np.load(f'explanations/{ds_name}/{ds_index}/ensemble_expl.npy')
    except Exception:
        rng = np.random.RandomState(20231110 + ds_index)
        indices = rng.choice(np.arange(len(x_train)), size=min(1000, len(x_train)), replace=False)
        background = x_train[indices]

        # If available, use gpu acceleration
        pt_xval_lin = torch.from_numpy(x_val[lin_indices])
        pt_yval_lin = torch.from_numpy(y_val[lin_indices])

        pt_xval_ens = torch.from_numpy(x_val[ensemble_indices])
        pt_yval_ens = torch.from_numpy(y_val[ensemble_indices])

        pt_background = torch.from_numpy(background)
        if torch.cuda.is_available():
            before = time.time()
            pt_xval_lin = pt_xval_lin.to('cuda')
            pt_xval_ens = pt_xval_ens.to('cuda')
            pt_yval_lin = pt_yval_lin.to('cuda')
            pt_yval_ens = pt_yval_ens.to('cuda')
            pt_background = pt_background.to('cuda')
            
            for estimator in pt_ensemble.ensemble:
                estimator.to('cuda')

            pt_linear.lin.to('cuda')
            after = time.time()
            print('x', pt_xval_lin.shape, pt_xval_ens.shape, 'y', pt_yval_lin.shape, pt_yval_ens.shape, 'bg', pt_background.shape)
            print('copying took', after-before, 'seconds')


        with fixedseed([torch, np], seed=(20231110+ds_index)):
            lin_expl = get_explanations(pt_linear, pt_xval_lin, pt_yval_lin, pt_background)
            ensemble_expl = get_explanations(pt_ensemble, pt_xval_ens, pt_yval_ens, pt_background)
        np.save(f'explanations/{ds_name}/{ds_index}/lin_expl.npy', lin_expl)
        np.save(f'explanations/{ds_name}/{ds_index}/ensemble_expl.npy', ensemble_expl)

    # Load (or compute) rocs
    if x_val[lin_indices].shape != lin_expl.shape:
        if x_val[lin_indices].shape[0] == 0:
            lin_rocs = []
        else:
            print('redo', ds_index, x_val[lin_indices].shape, lin_expl.shape)
            return val_results, test_results, selection_results
    else:
        lin_rocs = compute_rocs(x_val[lin_indices], y_val[lin_indices], lin_expl, lin_error[lin_indices])

    if x_val[ensemble_indices].shape != ensemble_expl.shape:
        if x_val[ensemble_indices].shape[0] == 0:
            ensemble_rocs = []
        else:
            print('redo', ds_index, x_val[ensemble_indices].shape, ensemble_expl.shape)
            return val_results, test_results, selection_results
    else:
        ensemble_rocs = compute_rocs(x_val[ensemble_indices], y_val[ensemble_indices], ensemble_expl, ensemble_error[ensemble_indices])
    
    # makedirs(f'plots/{ds_name}', exist_ok=True)
    # plot_rocs(lin_rocs, ensemble_rocs, f'plots/{ds_name}/{ds_index}.pdf')

    # ------------------ Run new selection methods
    
    # # v1
    # name, test_selection = run_v1(x_test, lin_rocs, ensemble_rocs)
    # test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
    # loss_test_test = rmse(test_prediction_test, y_test)
    # test_results[name] = loss_test_test
    # selection_results[name] = np.mean(test_selection)

    # # v4
    name, test_selection = run_v4(optimal_selection_val, x_val, x_test, lin_preds_val, nn_preds_val, lin_preds_test, nn_preds_test, lin_rocs, ensemble_rocs, random_state=20231116+ds_index)
    test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
    loss_test_test = rmse(test_prediction_test, y_test)
    test_results[name] = loss_test_test
    selection_results[name] = np.mean(test_selection)

    name, test_selection = run_v4(optimal_selection_val, x_val, x_test, lin_preds_val, nn_preds_val, lin_preds_test, nn_preds_test, lin_rocs, ensemble_rocs, random_state=20231116+ds_index, calibrate=True)
    test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
    loss_test_test = rmse(test_prediction_test, y_test)
    test_results[name] = loss_test_test
    selection_results[name] = np.mean(test_selection)

    # makedirs(f'test/{ds_name}/{ds_index}', exist_ok=True)
    # basepath = f'test/{ds_name}/{ds_index}'
    # from os.path import join
    # np.save(join(basepath, 'x_val.npy'), x_val)
    # np.save(join(basepath, 'y_val.npy'), y_val)
    # np.save(join(basepath, 'x_test.npy'), x_test)
    # np.save(join(basepath, 'y_test.npy'), y_test)
    # np.save(join(basepath, 'lin_preds_val.npy'), lin_preds_val)
    # np.save(join(basepath, 'nn_preds_val.npy'), nn_preds_val)
    # np.save(join(basepath, 'lin_preds_test.npy'), lin_preds_test)
    # np.save(join(basepath, 'nn_preds_test.npy'), nn_preds_test)
    # with open(join(basepath, 'lin_rocs.pickle'), 'wb+') as f:
    #     pickle.dump(lin_rocs, f)
    # with open(join(basepath, 'ensemble_rocs.pickle'), 'wb+') as f:
    #     pickle.dump(ensemble_rocs, f)

    # return test_results, val_results, selection_results

    # v5
    # name, test_selection = run_v5(x_val, y_val, x_test, y_test, lin_preds_val, nn_preds_val, lin_preds_test, nn_preds_test, lin_rocs, ensemble_rocs, random_state=20231116+ds_index, p=0.99)
    # test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
    # loss_test_test = rmse(test_prediction_test, y_test)
    # test_results[name] = loss_test_test
    # selection_results[name] = np.mean(test_selection)

    # name, test_selection = run_v5(x_val, y_val, x_test, y_test, lin_preds_val, nn_preds_val, lin_preds_test, nn_preds_test, lin_rocs, ensemble_rocs, random_state=20231116+ds_index, p=0.95)
    # test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
    # loss_test_test = rmse(test_prediction_test, y_test)
    # test_results[name] = loss_test_test
    # selection_results[name] = np.mean(test_selection)

    # name, test_selection = run_v5(x_val, y_val, x_test, y_test, lin_preds_val, nn_preds_val, lin_preds_test, nn_preds_test, lin_rocs, ensemble_rocs, random_state=20231116+ds_index, p=0.90)
    # test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
    # loss_test_test = rmse(test_prediction_test, y_test)
    # test_results[name] = loss_test_test
    # selection_results[name] = np.mean(test_selection)

    # name, test_selection = run_v6(x_test, lin_rocs, ensemble_rocs)
    # test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
    # loss_test_test = rmse(test_prediction_test, y_test)
    # test_results[name] = loss_test_test
    # selection_results[name] = np.mean(test_selection)

    return val_results, test_results, selection_results


if __name__ == '__main__':
    main()
