import tqdm
import numpy as np
import pickle
import pandas as pd

from config import DATASET_HYPERPARAMETERS
from preprocessing import load_local_data, create_selector_features
from tsx.model_selection import ADE, DETS, KNNRoC, OMS_ROC
from tsx.utils import string_to_randomstate
from utils import smape, rmse
from os import makedirs
from config import ALL_DATASETS

def _compute_individual(X_train, y_train, X_val, y_val, X_test, y_test, fcomp_preds_val, fcomp_preds_test, fint_preds_val, fint_preds_test, random_state=None):
    # TODO: Provide the same input data as in AALF

    results = {}
    val_preds = np.vstack([fcomp_preds_val, fint_preds_val])
    test_preds = np.vstack([fcomp_preds_test, fint_preds_test])

    try:
        ade = ADE(random_state)
        prediction, selection = ade.run(X_val, y_val, val_preds, X_test, y_test, test_preds, only_best=True, _omega=1)
        loss_rmse = rmse(y_test, prediction)
        loss_smape = smape(y_test, prediction)
        p = np.mean(selection)
        results = results | {'ade_rmse': loss_rmse, 'ade_smape': loss_smape, 'ade_p': p}
    except Exception as e:
        print('ERROR ADE')
        print(e)
        exit()

    try:
        knn = KNNRoC()
        prediction, selection = knn.run(X_val, y_val, val_preds, X_test, y_test, test_preds)
        loss_rmse = rmse(y_test, prediction)
        loss_smape = smape(y_test, prediction)
        p = np.mean(selection)
        results = results | {'knnroc_rmse': loss_rmse, 'knnroc_smape': loss_smape, 'knnroc_p': p}
    except Exception as e:
        print('ERROR KNN')
        print(e)
        exit()

    try:
        oms = OMS_ROC(nc_max=10, random_state=random_state)
        prediction, selection = oms.run(X_val, y_val, val_preds, X_test, y_test, test_preds)
        loss_rmse = rmse(y_test, prediction)
        loss_smape = smape(y_test, prediction)
        p = np.mean(selection)
        results = results | {'omsroc_rmse': loss_rmse, 'omsroc_smape': loss_smape, 'omsroc_p': p}
    except Exception as e:
        print('ERROR OMS_ROC')
        print(e)
        exit()

    try:
        dets = DETS()
        prediction, selection = dets.run(X_val, y_val, val_preds, X_test, y_test, test_preds, only_best=True)
        loss_rmse = rmse(y_test, prediction)
        loss_smape = smape(y_test, prediction)
        p = np.mean(selection)
        results = results | {'dets_rmse': loss_rmse, 'dets_smape': loss_smape, 'dets_p': p}
    except Exception as e:
        print('ERROR DETS')
        print(e)
        exit()

    return results

def compute_baselines(ds_name, debug=False):

    # Load dataset hyperparameters
    dsh = DATASET_HYPERPARAMETERS[ds_name] 
    L = dsh['L']
    if 'fint' not in dsh or 'fcomp' not in dsh:
        return
    fint_name = dsh['fint']
    fcomp_name = dsh['fcomp']

    # Load data
    (local_X_train, local_y_train), (local_X_val, local_y_val), (local_X_test, local_y_test) = load_local_data(ds_name, L=L, H=1, return_split=['train', 'val', 'test'], verbose=False)
    with open(f'preds/{ds_name}.pickle', 'rb') as f:
        preds = pickle.load(f)

    n_datapoints = len(local_X_train)
    seed = string_to_randomstate(ds_name, return_seed=True)

    # Prepare data for model selection methods
    fint_val_preds = preds['val'][fint_name]
    fint_test_preds = preds['test'][fint_name]
    fcomp_val_preds = preds['val'][fcomp_name]
    fcomp_test_preds = preds['test'][fcomp_name]

    # Run selections
    if n_datapoints > 16 and not debug:
        from joblib import Parallel, delayed
        result = Parallel(n_jobs=-1, backend='loky')(delayed(_compute_individual)(
            X_train=local_X_train[ds_index],
            y_train=local_y_train[ds_index],
            X_val=local_X_val[ds_index],
            X_test=local_X_test[ds_index],
            y_val=local_y_val[ds_index],
            y_test=local_y_test[ds_index],
            fint_preds_val=fint_val_preds[ds_index],
            fint_preds_test=fint_test_preds[ds_index],
            fcomp_preds_val=fcomp_val_preds[ds_index],
            fcomp_preds_test=fcomp_test_preds[ds_index],
            random_state=seed) for ds_index in tqdm.trange(n_datapoints, desc=f'[{ds_name}]'))
    else:
        result = [_compute_individual(
            X_train=local_X_train[ds_index],
            y_train=local_y_train[ds_index],
            X_val=local_X_val[ds_index],
            X_test=local_X_test[ds_index],
            y_val=local_y_val[ds_index],
            y_test=local_y_test[ds_index],
            fint_preds_val=fint_val_preds[ds_index],
            fint_preds_test=fint_test_preds[ds_index],
            fcomp_preds_val=fcomp_val_preds[ds_index],
            fcomp_preds_test=fcomp_test_preds[ds_index],
            random_state=seed) for ds_index in tqdm.trange(n_datapoints, desc=f'[{ds_name}-DBG]')]

    result = pd.DataFrame(result)

    # Save results
    if not debug:
        makedirs('results/baseline_selectors/', exist_ok=True)
        result.to_csv(f'results/baseline_selectors/{ds_name}.csv')

def main():
    for ds_name in ALL_DATASETS:
        compute_baselines(ds_name, debug=False)

if __name__ == '__main__':
    main()