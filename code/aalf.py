import pandas as pd
import numpy as np
import tqdm
import pickle 
from selection import Oracle
from preprocessing import load_local_data, create_selector_features
from config import DATASET_HYPERPARAMETERS, ALL_DATASETS
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from os import makedirs
from utils import rmse, smape

class AALF:

    def __init__(self, p):
        self.p = p

    def run(self, X_val, y_val, val_preds, X_test, y_test, test_preds):

        # Get label to train local selector
        oracle = Oracle(self.p)
        s_star_val = oracle.get_labels(y_val, val_preds[0], val_preds[1])

        sel = RandomForestClassifier(n_estimators=128, random_state=20241127)
        sel.fit(X_val, s_star_val)
        selection = sel.predict(X_test).astype(np.int8)

        prediction = np.choose(selection, test_preds)
        return prediction, selection


def _run_single(p, X_train, y_train, X_val, y_val, X_test, y_test, fcomp_preds_train, fcomp_preds_val, fcomp_preds_test, fint_preds_train, fint_preds_val, fint_preds_test):
    train_preds = np.vstack([fcomp_preds_train, fint_preds_train])
    val_preds = np.vstack([fcomp_preds_val, fint_preds_val])
    test_preds = np.vstack([fcomp_preds_test, fint_preds_test])

    _X_val = create_selector_features(X_train, y_train, X_val, y_val, train_preds, val_preds)
    _X_test = create_selector_features(X_val, y_val, X_test, y_test, val_preds, test_preds)

    aalf = AALF(p)
    prediction, selection = aalf.run(_X_val, y_val, val_preds, _X_test, y_test, test_preds)
    loss_rmse = rmse(y_test, prediction)
    loss_smape = smape(y_test, prediction)
    _p = np.mean(selection)
    results = {'aalf_rmse': loss_rmse, 'aalf_smape': loss_smape, 'aalf_p': _p, 'true_p': p}

    return results

def run(ds_name, p, debug=False):
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

    # Prepare data for model selection methods
    fint_train_preds = preds['train'][fint_name]
    fint_val_preds = preds['val'][fint_name]
    fint_test_preds = preds['test'][fint_name]
    fcomp_train_preds = preds['train'][fcomp_name]
    fcomp_val_preds = preds['val'][fcomp_name]
    fcomp_test_preds = preds['test'][fcomp_name]

    # Run selections
    if n_datapoints > 16 and not debug:
        from joblib import Parallel, delayed
        result = Parallel(n_jobs=-1, backend='loky')(delayed(_run_single)(
            p=p,
            X_train=local_X_train[ds_index],
            y_train=local_y_train[ds_index],
            X_val=local_X_val[ds_index],
            X_test=local_X_test[ds_index],
            y_val=local_y_val[ds_index],
            y_test=local_y_test[ds_index],
            fint_preds_train=fint_train_preds[ds_index],
            fint_preds_val=fint_val_preds[ds_index],
            fint_preds_test=fint_test_preds[ds_index],
            fcomp_preds_train=fcomp_train_preds[ds_index],
            fcomp_preds_val=fcomp_val_preds[ds_index],
            fcomp_preds_test=fcomp_test_preds[ds_index],
            ) for ds_index in tqdm.trange(n_datapoints, desc=f'[{ds_name} - {p}]'))
    else:
        result = [_run_single(
            p=p,
            X_train=local_X_train[ds_index],
            y_train=local_y_train[ds_index],
            X_val=local_X_val[ds_index],
            X_test=local_X_test[ds_index],
            y_val=local_y_val[ds_index],
            y_test=local_y_test[ds_index],
            fint_preds_train=fint_train_preds[ds_index],
            fint_preds_val=fint_val_preds[ds_index],
            fint_preds_test=fint_test_preds[ds_index],
            fcomp_preds_train=fcomp_train_preds[ds_index],
            fcomp_preds_val=fcomp_val_preds[ds_index],
            fcomp_preds_test=fcomp_test_preds[ds_index],
            ) for ds_index in tqdm.trange(n_datapoints, desc=f'[{ds_name} - {p} - DBG]')]

    result = pd.DataFrame(result)
    makedirs('results/aalf', exist_ok=True)
    result.to_csv(f'results/aalf/{ds_name}_{p}.csv')

def main():
    for ds_name in ALL_DATASETS:
        for p in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            run(ds_name, p=p, debug=False)

if __name__ == '__main__':
    main()