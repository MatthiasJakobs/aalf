import torch
import tqdm
import numpy as np
import pandas as pd
import pickle

from seedpy import fixedseed
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from seedpy import fixedseed
from os import makedirs
from joblib import Parallel, delayed
from os.path import exists
from tsx.model_selection import ADE, DETS, KNNRoC, OMS_ROC

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from tsx.models.forecaster.baselines import TableForestRegressor

from selection import run_v12, oracle
from datasets import load_dataset
from models import MedianPredictionEnsemble, HetEnsemble, fit_basemodels, CNN
from evaluation import load_models, preprocess_data
from utils import rmse
from config import Ls

def main(to_run, njobs, debug=False):

    ds_names = ['nn5']

    if to_run is None:
        to_run = []

    for ds_name in ds_names:
        log_test = []
        log_selection = []

        if debug:
            fraction = 0.1
        else:
            fraction = 1

        L = Ls[ds_name]
        X = load_dataset(ds_name, fraction=fraction)

        print(ds_name, 'n_series', len(X))

        # Load previous results (if available), otherwise create empty
        if exists(f'results/{ds_name}_test.csv'):
            test_results = pd.read_csv(f'results/{ds_name}_test.csv')
            test_results = test_results.set_index('dataset_names')
            test_results = test_results.T.to_dict()
        else:
            test_results = [{} for _ in range(len(X))]

        if exists(f'results/{ds_name}_selection.csv'):
            test_selection = pd.read_csv(f'results/{ds_name}_selection.csv')
            test_selection = test_selection.set_index('dataset_names')
            test_selection = test_selection.T.to_dict()
        else:
            test_selection = [{} for _ in range(len(X))]

        if exists(f'results/{ds_name}_gfi.csv'):
            test_gfi = pd.read_csv(f'results/{ds_name}_gfi.csv')
            test_gfi = test_gfi.set_index('dataset_names')
            test_gfi = test_gfi.T.to_dict()
        else:
            test_gfi = [{} for _ in range(len(X))]

        print('To run', to_run)

        log_test, log_selection, log_gfi = zip(*Parallel(n_jobs=njobs, backend='loky')(delayed(run_experiment)(ds_name, ds_index, X[ds_index], L, 1, test_results[ds_index], test_selection[ds_index], test_gfi[ds_index], to_run=to_run) for ds_index in tqdm.trange(len(X))))

        makedirs('results', exist_ok=True)
        log_test = pd.DataFrame(list(log_test))
        log_test = log_test.set_index('dataset_names')
        log_test.to_csv(f'results/{ds_name}_test.csv')

        log_selection = pd.DataFrame(list(log_selection))
        log_selection = log_selection.set_index('dataset_names')
        log_selection.to_csv(f'results/{ds_name}_selection.csv')

        log_gfi = pd.DataFrame(list(log_gfi))
        log_gfi = log_gfi.set_index('dataset_names')
        log_gfi.to_csv(f'results/{ds_name}_gfi.csv')

def run_experiment(ds_name, ds_index, X, L, H, test_results, selection_results, gfi_results, lr=1e-3, max_iter_nn=500, to_run=None):
    makedirs(f'models/{ds_name}/{ds_index}', exist_ok=True)

    test_results['dataset_names'] = ds_index
    selection_results['dataset_names'] = ds_index
    gfi_results['dataset_names'] = ds_index

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocess_data(X, L, H)

    if (x_train.reshape(-1)[0] == x_train).all() or (x_val.reshape(-1)[0] == x_val).all():
        print('to remove', ds_name, ds_index)
        return test_results, selection_results, gfi_results

    # if exists(f'models/{ds_name}/{ds_index}/linear.pickle') and exists(f'models/{ds_name}/{ds_index}/nns/0.pickle'):
    #     f_i, f_c = load_models(ds_name, ds_index)
    # else:
    #     f_i = LinearRegression()
    #     f_i.fit(x_train, y_train)
    #     with open(f'models/{ds_name}/{ds_index}/linear.pickle', 'wb') as f:
    #         pickle.dump(f_i, f)

    #     with fixedseed(np, 20231103):
    #         f_c = MedianPredictionEnsemble([MLPRegressor((28,), learning_rate_init=lr, max_iter=max_iter_nn) for _ in range(10)])
    #         f_c.fit(x_train, y_train.squeeze())
    #         f_c.save_model(ds_name, ds_index)

    f_i = LinearRegression()
    f_i.fit(x_train, y_train)

    #makedirs(f'preds/{ds_name}/{ds_index}', exist_ok=True)

    lin_preds_train = f_i.predict(x_train)
    lin_preds_val = f_i.predict(x_val)
    lin_preds_test = f_i.predict(x_test)
    loss_i_test = rmse(lin_preds_test, y_test)
    #np.save(f'preds/{ds_name}/{ds_index}/lin.npy', lin_preds_test.reshape(-1))
    test_results['linear'] = loss_i_test

    if not 'trf-raw' in test_results:
        rf = TableForestRegressor(n_estimators=128, include_raw=True, random_state=19278161)
        rf.fit(x_train, y_train.squeeze())
        rf_preds_test = rf.predict(x_test)
        loss_test = rmse(rf_preds_test, y_test)
        test_results['trf-raw'] = loss_test

    # nn_preds_train = f_c.predict(x_train)
    # nn_preds_val = f_c.predict(x_val)
    # nn_preds_test = f_c.predict(x_test)
    # loss_nn_test = rmse(nn_preds_test, y_test)
    # np.save(f'preds/{ds_name}/{ds_index}/nn.npy', nn_preds_test.reshape(-1))
    # test_results['nn'] = loss_nn_test

    # closeness_threshold = 1e-6

    # lin_val_error = (lin_preds_val.squeeze()-y_val.squeeze())**2
    # nn_val_error = (nn_preds_val.squeeze()-y_val.squeeze())**2
    # lin_val_better = np.where(lin_val_error-nn_val_error <= closeness_threshold)[0]
    # nn_val_better = np.array([idx for idx in range(len(lin_val_error)) if idx not in lin_val_better]).astype(np.int32)
    # assert len(lin_val_better) + len(nn_val_better) == len(lin_val_error)

    # lin_test_error = (lin_preds_test.squeeze()-y_test.squeeze())**2
    # nn_test_error = (nn_preds_test.squeeze()-y_test.squeeze())**2
    # lin_test_better = np.where(lin_test_error-nn_test_error <= closeness_threshold)[0]
    # nn_test_better = np.array([idx for idx in range(len(lin_test_error)) if idx not in lin_test_better]).astype(np.int32)
    # assert len(lin_test_better) + len(nn_test_better) == len(lin_test_error)

    # -------------------------------- Model selection

    # lin_preds_val = lin_preds_val.squeeze()
    # lin_preds_test = lin_preds_test.squeeze()
    # nn_preds_val = nn_preds_val.squeeze()
    # nn_preds_test = nn_preds_test.squeeze()
    # y_val = y_val.squeeze()
    # y_test = y_test.squeeze()

    # optimal_selection_test = (lin_test_error - nn_test_error) <= closeness_threshold

    # optimal_prediction_test = np.choose(optimal_selection_test, [nn_preds_test, lin_preds_test])
    # loss_optimal_test = rmse(optimal_prediction_test, y_test)
    # test_results['selOpt'] = loss_optimal_test
    # selection_results['selOpt'] = np.mean(optimal_selection_test)

    # - Model selection

    # if 'v12' in to_run or 'v12_0.5' not in test_results.keys():
    #     for p in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
    #         name, test_selection, gfi = run_v12(lin_preds_train, nn_preds_train, y_train, y_test, x_val, y_val, x_test, lin_preds_val, nn_preds_val, lin_preds_test, nn_preds_test, random_state=20231322+ds_index, p=p)
    #         if p == 0.9 and gfi is not None:
    #             for feat_idx in range(12):
    #                 gfi_results[feat_idx] = gfi[feat_idx]
    #         test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
    #         loss_test_test = rmse(test_prediction_test, y_test)
    #         np.save(f'preds/{ds_name}/{ds_index}/{name}.npy', test_prediction_test.reshape(-1))
    #         test_results[name] = loss_test_test
    #         selection_results[name] = np.mean(test_selection)

    # if 'baselines' in to_run or 'ade' not in test_results.keys():
    #     name = 'ade'
    #     ade = ADE(20240212+ds_index)
    #     try:
    #         _, test_selection = ade.run(x_val, y_val.squeeze(), np.vstack([nn_preds_val, lin_preds_val]), x_test, y_test.squeeze(), np.vstack([nn_preds_test, lin_preds_test]), only_best=True, _omega=1)
    #     except Exception as e:
    #         print('ERROR ade', ds_index)
    #         print(e)
    #         exit()
    #     test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
    #     loss_test_test = rmse(test_prediction_test, y_test)
    #     np.save(f'preds/{ds_name}/{ds_index}/{name}.npy', test_prediction_test.reshape(-1))
    #     test_results[name] = loss_test_test
    #     selection_results[name] = np.mean(test_selection)

    #     name = 'knnroc'
    #     knn = KNNRoC([f_c, f_i])
    #     try:
    #         _, test_selection = knn.run(x_val, y_val, x_test)
    #     except Exception as e:
    #         print('ERROR knnroc', ds_index)
    #         print(e)
    #         exit()
    #     test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
    #     loss_test_test = rmse(test_prediction_test, y_test)
    #     np.save(f'preds/{ds_name}/{ds_index}/{name}.npy', test_prediction_test.reshape(-1))
    #     test_results[name] = loss_test_test
    #     selection_results[name] = np.mean(test_selection)

    #     name = 'oms'
    #     oms = OMS_ROC([f_c, f_i], nc_max=10, random_state=20240212+ds_index)
    #     try:
    #         _, test_selection = oms.run(x_val, y_val, x_test)
    #     except Exception as e:
    #         print('ERROR oms', ds_index)
    #         print(e)
    #         exit()
    #     test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
    #     loss_test_test = rmse(test_prediction_test, y_test)
    #     np.save(f'preds/{ds_name}/{ds_index}/{name}.npy', test_prediction_test.reshape(-1))
    #     test_results[name] = loss_test_test
    #     selection_results[name] = np.mean(test_selection)

    #     name = 'dets'
    #     dets = DETS()
    #     try:
    #         _, test_selection = dets.run(x_val, y_val.squeeze(), np.vstack([nn_preds_val, lin_preds_val]), x_test, y_test.squeeze(), np.vstack([nn_preds_test, lin_preds_test]), only_best=True)
    #     except AssertionError:
    #         print('ERROR dets', ds_index)
    #         exit()
    #     test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
    #     loss_test_test = rmse(test_prediction_test, y_test)
    #     np.save(f'preds/{ds_name}/{ds_index}/{name}.npy', test_prediction_test.reshape(-1))
    #     test_results[name] = loss_test_test
    #     selection_results[name] = np.mean(test_selection)

    # for p in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
    #     name = f'NewOracle{int(100*p)}'
    #     test_selection = oracle(lin_preds_test, nn_preds_test, y_test, p)
    #     test_prediction_test = np.choose(test_selection, [nn_preds_test, lin_preds_test])
    #     loss_test_test = rmse(test_prediction_test, y_test)
    #     np.save(f'preds/{ds_name}/{ds_index}/{name}.npy', test_prediction_test.reshape(-1))
    #     test_results[name] = loss_test_test
    #     selection_results[name] = np.mean(test_selection)
    
    # basemodels = fit_basemodels(ds_name, ds_index, x_train, y_train, random_state=20241014)
    # for m_name, m in basemodels.items():
    #     test_prediction_test = m.predict(x_test).reshape(-1)
    #     loss_test_test = rmse(test_prediction_test, y_test.reshape(-1))
    #     test_results[m_name] = loss_test_test

    # ens = HetEnsemble(list(basemodels.values()))
    # test_prediction_test = ens.predict(x_test).reshape(-1)
    # loss_test_test = rmse(test_prediction_test, y_test.reshape(-1))
    # test_results['HetEnsemble'] = loss_test_test

    # if not exists(f'models/{ds_name}/{ds_index}/cnn.pickle'):
    #     with fixedseed(torch, 20241012):
    #         cnn = CNN(L)
    #         cnn.fit(x_train, y_train.squeeze())
    #     with open(f'models/{ds_name}/{ds_index}/cnn.pickle', 'wb') as f:
    #         pickle.dump(cnn, f)
    # else:
    #     with open(f'models/{ds_name}/{ds_index}/cnn.pickle', 'rb') as f:
    #         cnn = pickle.load(f)

    # test_prediction_test = cnn.predict(x_test).reshape(-1)
    # loss_test_test = rmse(test_prediction_test, y_test.reshape(-1))
    # test_results['cnn'] = loss_test_test

    with open(f'models/cnn_{ds_name}.pickle', 'rb') as f:
        global_cnn = pickle.load(f)
    test_prediction_test = global_cnn.predict(x_test).reshape(-1)
    loss_test_test = rmse(test_prediction_test, y_test.reshape(-1))
    test_results['global_cnn'] = loss_test_test

    name = 'LastValue'
    test_prediction_test = x_test[:, -1].reshape(-1)
    loss_test_test = rmse(test_prediction_test, y_test)
    #np.save(f'preds/{ds_name}/{ds_index}/{name}.npy', test_prediction_test)
    test_results[name] = loss_test_test

    name = 'MeanValue'
    test_prediction_test = x_test.mean(axis=1).reshape(-1)
    loss_test_test = rmse(test_prediction_test, y_test)
    #np.save(f'preds/{ds_name}/{ds_index}/{name}.npy', test_prediction_test)
    test_results[name] = loss_test_test

    return test_results, selection_results, gfi_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--override", help='', nargs='+', default=[])
    parser.add_argument("--debug", help='', type=bool, default=False)
    parser.add_argument("--njobs", help='', type=int, default=-1)
    args = vars(parser.parse_args())
    main(args['override'], args['njobs'], debug=args['debug'])
