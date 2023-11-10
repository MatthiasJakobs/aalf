import skorch
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

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

from cdd_plots import create_cdd
from models import Ensemble, PyTorchLinear, PyTorchEnsemble
from explainability import get_explanations
from tsx.model_selection import ROC_Member
from tsx.models.forecaster import split_zero

def rmse(a, b):
    return mean_squared_error(a, b, squared=False)

def compute_rocs(x, y, explanations, threshold=0):
    # Threshold and invert explanations 
    explanations =  explanations / explanations.sum(axis=1).reshape(-1, 1)
    explanations = -explanations * ((-explanations) > threshold)

    rocs = []
    for i, e in enumerate(explanations):
        splits = split_zero(e, min_size=3)
        for (f, t) in splits:
            r = ROC_Member(x[i], y[i], np.arange(t-f+1)+f)
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
    for i, x in enumerate(X):
        print('-'*30, i, '-'*30)
        log_val, log_test = run_experiment(log_val, log_test, 'weather', i, x, L, horizons[i], verbose=True)

    log_val = pd.DataFrame(log_val)
    log_val.index.rename('dataset_names', inplace=True)
    log_val.to_csv('results/weather_val.csv')
    log_test = pd.DataFrame(log_test)
    log_test.index.rename('dataset_names', inplace=True)
    log_test.to_csv('results/weather_test.csv')

    create_cdd('weather')

def run_experiment(log_val, log_test, ds_name, ds_index, X, L, H, lr=1e-3, verbose=False):
    print(H, 'ts length', X.shape)
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

    # Random Forests
    try:
        with open(f'models/{ds_name}/{ds_index}/rf.pickle', 'rb') as f:
            f_c = pickle.load(f)
    except Exception:
        f_c = RandomForestRegressor(n_estimators=128, max_depth=8, random_state=12345, n_jobs=-1)
        f_c.fit(x_train, y_train.squeeze())
        with open(f'models/{ds_name}/{ds_index}/rf.pickle', 'wb') as f:
            pickle.dump(f_c, f)
        
    loss_rf_val = rmse(f_c.predict(x_val).squeeze(), y_val)
    loss_rf_test = rmse(f_c.predict(x_test).squeeze(), y_test)
    val_results['rf'] = loss_rf_val
    test_results['rf'] = loss_rf_test

    # Neural net
    try:
        with open(f'models/{ds_name}/{ds_index}/nn.pickle', 'rb') as f:
            f_c = pickle.load(f)
    except Exception:
        with fixedseed(np, 20231103):
            f_c = Ensemble(MLPRegressor, 10, (28,), learning_rate_init=lr, max_iter=500)
            f_c.fit(x_train, y_train.squeeze())
        with open(f'models/{ds_name}/{ds_index}/nn.pickle', 'wb') as f:
            pickle.dump(f_c, f)

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
    print('optimal selection val', np.mean(optimal_selection_val))
    print('optimal selection test', np.mean(optimal_selection_test))

    optimal_prediction_val = np.choose(optimal_selection_val, [nn_preds_val, lin_preds_val])
    optimal_prediction_test = np.choose(optimal_selection_test, [nn_preds_test, lin_preds_test])
    loss_optimal_val = rmse(optimal_prediction_val, y_val)
    loss_optimal_test = rmse(optimal_prediction_test, y_test)
    val_results['selOpt'] = loss_optimal_val
    test_results['selOpt'] = loss_optimal_test

    # -------------------------------- Threshold baseline
    for thresh in [0.5, 0.99, 1.5, 1.7]:
        thresh_selection_val = (lin_preds_val-nn_preds_val)**2 / (lin_preds_val**2 + nn_preds_val**2) <= thresh
        thresh_selection_test = (lin_preds_test-nn_preds_test)**2 / (lin_preds_test**2 + nn_preds_test**2) <= thresh
        # thresh_selection_val = (lin_preds_val-nn_preds_val)**2 <= thresh
        # thresh_selection_test = (lin_preds_test-nn_preds_test)**2 <= thresh
        print(f'{thresh} thresh selection val', np.mean(thresh_selection_val))
        print(f'{thresh} thresh selection test', np.mean(thresh_selection_test))

        thresh_prediction_val = np.choose(thresh_selection_val, [nn_preds_val, lin_preds_val])
        thresh_prediction_test = np.choose(thresh_selection_test, [nn_preds_test, lin_preds_test])
        loss_thresh_val = rmse(thresh_prediction_val, y_val)
        loss_thresh_test = rmse(thresh_prediction_test, y_test)
        val_results[f'selThresh{thresh}'] = loss_thresh_val
        test_results[f'selThresh{thresh}'] = loss_thresh_test

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
    print('percentage linear chosen', selection.mean())
    
    # Compute explanations on validation data points
    try:
        lin_expl = np.load(f'explanations/{ds_name}/{ds_index}/lin_expl.npy')
        ensemble_expl = np.load(f'explanations/{ds_name}/{ds_index}/ensemble_expl.npy')
    except Exception:
        rng = np.random.RandomState(20231110 + ds_index)
        indices = rng.choice(np.arange(len(x_train)), size=min(1000, len(x_train)), replace=False)
        background = x_train[indices]

        with fixedseed([torch, np], seed=(20231110+ds_index)):
            lin_expl = get_explanations(pt_linear, torch.from_numpy(x_val[lin_indices]), torch.from_numpy(y_val[lin_indices]), torch.from_numpy(background))
            ensemble_expl = get_explanations(pt_ensemble, torch.from_numpy(x_val[ensemble_indices]), torch.from_numpy(y_val[ensemble_indices]), torch.from_numpy(background))
        np.save(f'explanations/{ds_name}/{ds_index}/lin_expl.npy', lin_expl)
        np.save(f'explanations/{ds_name}/{ds_index}/ensemble_expl.npy', ensemble_expl)

    # Threshold negative loss attributions
    thresh = 0

    lin_rocs = compute_rocs(x_val[lin_indices], y_val[lin_indices], lin_expl)
    ensemble_rocs = compute_rocs(x_val[ensemble_indices], y_val[ensemble_indices], ensemble_expl)
    print('nr lin RoCs', len(lin_rocs))
    print('nr ensemble RoCs', len(ensemble_rocs))

    log_val.append(val_results)
    log_test.append(test_results)

    # Run experiments
    test_selection = []
    sel_threshold = 5
    for _x in x_test:
        lin_min_dist = min([lin_roc.euclidean_distance(_x) for lin_roc in lin_rocs])
        ensemble_min_dist = min([ensemble_roc.euclidean_distance(_x) for ensemble_roc in ensemble_rocs])
        if lin_min_dist <= ensemble_min_dist:
            test_selection.append(1)
        else:
            test_selection.append(0)
        #     continue
        # if ensemble_min_dist-lin_min_dist < sel_threshold:
        #     test_selection.append(1)
        # else:
        #     test_selection.append(0)

    test_selection = np.array(test_selection)
    print('new selection percent linear', np.mean(test_selection))
    test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
    loss_test_test = rmse(test_prediction_test, y_test)
    test_results[f'new'] = loss_test_test

    return log_val, log_test


if __name__ == '__main__':
    main()
