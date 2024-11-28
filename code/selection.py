import tqdm
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from utils import rmse, load_dict_or_create
from plotz import default_plot, COLORS
from config import DS_MAP, DATASET_HYPERPARAMETERS
from preprocessing import load_local_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from os import makedirs
from models import RandomSelector, UpsampleEnsembleClassifier
from tsx.utils import string_to_randomstate

SELECTORS = {
    'random_selector': {
        'model_class': RandomSelector,
        'hyperparameters': {},
        'n_repeats': 100,
        'randomized': True,
    },
    'logistic_regression': {
        'model_class': LogisticRegression,
        'hyperparameters': {},
    },
    'random_forest_128': {
        'model_class': RandomForestClassifier,
        'hyperparameters': {'n_estimators': 128, 'random_state': 20241127},
        'randomized': True,
    },
    'upsample_logistic_regression': {
        'model_class': UpsampleEnsembleClassifier,
        'hyperparameters': {'model_class': LogisticRegression, 'n_member': 64, 'random_state': 20241127},
        'randomized': True,
    },
    'upsample_tree': {
        'model_class': UpsampleEnsembleClassifier,
        'hyperparameters': {'model_class': DecisionTreeClassifier, 'n_member': 128, 'random_state': 20241127},
        'randomized': True,
    },
    'upsample_forest': {
        'model_class': UpsampleEnsembleClassifier,
        'hyperparameters': {'model_class': RandomForestClassifier, 'n_member': 9, 'n_estimators': 128, 'random_state': 20241127},
        'randomized': True,
    },
}

class Oracle:

    def __init__(self, p, threshold=1e-5):
        self.p = p
        self.threshold = threshold

    def check_shapes(self, a, b, c):
        shapes = [x.shape for x in [a, b, c]]
        if not all([len(shape) == len(shapes[0]) for shape in shapes]): 
            raise RuntimeError('Number of dimensions mismatch:', a.shape, b.shape, c.shape)
        shape_a = a.shape
        if not all([s == shape_a for s in shapes]):
            raise RuntimeError('Shape mismatch:', a.shape, b.shape, c.shape)

    # Return 0 if preds_a is better and 1 if preds_b is better
    def get_labels(self, y_true, preds_fcomp, preds_fint):
        preds_fcomp = preds_fcomp.squeeze()
        preds_fint = preds_fint.squeeze()
        y_true = y_true.squeeze()

        self.check_shapes(preds_fcomp, preds_fint, y_true)

        errors_fcomp = (preds_fcomp - y_true)**2
        errors_fint = (preds_fint - y_true)**2

        # Positive if b is better than a
        errors = (errors_fint-errors_fcomp)

        # How many to take at least
        B0 = int(np.ceil(self.p * len(errors)))
        # How many to take at max
        Bmax = (errors < self.threshold).sum()

        B = max(Bmax, B0)

        label = np.zeros((len(errors)))
        label[np.argsort(errors)[:B]] = 1

        return label.astype(np.int8)

def get_last_errors(fint_preds_val, fcomp_preds_val, y_val, fint_preds_test, fcomp_preds_test, y_test):
    last_preds_fint = np.concatenate([fint_preds_val[-1].reshape(-1), fint_preds_test[:-1].reshape(-1)])
    last_preds_fcomp = np.concatenate([fcomp_preds_val[-1].reshape(-1), fcomp_preds_test[:-1].reshape(-1)])
    y_true = np.concatenate([y_val[-1].reshape(-1), y_test[:-1].reshape(-1)])
    return (last_preds_fint-y_true)**2 - (last_preds_fcomp-y_true)**2

def compute_selector(ds_name, fint_name, fcomp_name, sel_name, ps):
    # Load data and predictions
    dsh = DATASET_HYPERPARAMETERS[ds_name] 
    L = dsh['L']

    (local_X_train, local_y_train), (local_X_val, local_y_val), (local_X_test, local_y_test) = load_local_data(ds_name, L=L, H=1, return_split=['train', 'val', 'test'], verbose=False)
    with open(f'preds/{ds_name}.pickle', 'rb') as f:
        preds = pickle.load(f)

    makedirs('results/selection', exist_ok=True)
    results = load_dict_or_create(f'results/selection/{ds_name}.pickle')

    # Train local selectors
    model_class = SELECTORS[sel_name]['model_class']
    hyperparameters = SELECTORS[sel_name]['hyperparameters']
    n_repeats = SELECTORS[sel_name].get('n_repeats', 1)
    n_datapoints  = len(local_X_val)
    if 'randomized' in SELECTORS[sel_name].keys():
        hyperparameters['random_state'] = string_to_randomstate(f'{ds_name}_{sel_name}', return_seed=True)
    
    for p in tqdm.tqdm(ps, desc=f'[{ds_name} - {sel_name}]'):
        tn, fp, fn, tp = 0, 0, 0, 0
        selection_percentage = []

        for ds_index in range(n_datapoints):
            X_train = local_X_train[ds_index]
            X_val = local_X_val[ds_index]
            X_test = local_X_test[ds_index]
            y_train = local_y_train[ds_index]
            y_val = local_y_val[ds_index]
            y_test = local_y_test[ds_index]

            fint_train_preds = preds['train'][fint_name][ds_index]
            fcomp_train_preds = preds['train'][fcomp_name][ds_index]
            fint_val_preds = preds['val'][fint_name][ds_index]
            fcomp_val_preds = preds['val'][fcomp_name][ds_index]
            fint_test_preds = preds['test'][fint_name][ds_index]
            fcomp_test_preds = preds['test'][fcomp_name][ds_index]

            oracle = Oracle(p)
            s_star_val = oracle.get_labels(y_val, fcomp_val_preds, fint_val_preds)
            s_star_test = oracle.get_labels(y_test, fcomp_test_preds, fint_test_preds)

            # Add more features to input
            val_prediction_difference = (fcomp_val_preds - fint_val_preds).reshape(-1, 1)
            val_aggregated_windows = np.concatenate([
                np.mean(X_val, axis=1, keepdims=True),
                np.min(X_val, axis=1, keepdims=True),
                np.max(X_val, axis=1, keepdims=True),
            ], axis=-1)
            val_last_errors = get_last_errors(fint_train_preds, fcomp_train_preds, y_train, fint_val_preds, fcomp_val_preds, y_val).reshape(-1, 1)
            X_val = np.concatenate([X_val, val_prediction_difference, val_aggregated_windows, val_last_errors], axis=-1)

            test_prediction_difference = (fcomp_test_preds - fint_test_preds).reshape(-1, 1)
            test_aggregated_windows = np.concatenate([
                np.mean(X_test, axis=1, keepdims=True),
                np.min(X_test, axis=1, keepdims=True),
                np.max(X_test, axis=1, keepdims=True),
            ], axis=-1)
            test_last_errors = get_last_errors(fint_val_preds, fcomp_val_preds, y_val, fint_test_preds, fcomp_test_preds, y_test).reshape(-1, 1)
            X_test = np.concatenate([X_test, test_prediction_difference, test_aggregated_windows, test_last_errors], axis=-1)

            model = model_class(**hyperparameters)
            for _ in range(n_repeats):
                model.fit(X_val, s_star_val)
                selector_preds = model.predict(X_test).squeeze()

                _tn, _fp, _fn, _tp = confusion_matrix(s_star_test, selector_preds).ravel()
                tn += _tn
                fp += _fp
                fn += _fn
                tp += _tp

                selection_percentage.append(np.mean(selector_preds))

        # Compute compound f1 score
        f1 = (2 * tp) / (2 * tp + fp + fn)

        try:
            results[sel_name]
        except KeyError:
            results[sel_name] = {}

        results[sel_name][p] = {
            'f1': f1,
            'p': np.mean(selection_percentage),
            'p_std': np.std(selection_percentage),
        }
    with open(f'results/selection/{ds_name}.pickle', 'wb') as f:
        pickle.dump(results, f)

def compute_selection_accuracy(ds_name, ps):
    with open(f'results/selection/{ds_name}.pickle', 'rb') as f:
        results = pickle.load(f)

    df = pd.DataFrame(columns=['model_name']+[str(p) for p in ps])
    model_names = list(results.keys())
    for model_name in model_names:
        scores = {str(k): [float(v['f1'])] for k, v in results[model_name].items()} | {'model_name': [model_name]}
        df = pd.concat([df, pd.DataFrame(scores)], ignore_index=True)

    df = df.set_index('model_name')
    print(df.loc[list(SELECTORS.keys())])
    return df
        
def main():
    ps = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    # nn5 daily
    ds_name = 'nn5_daily_nomissing'
    fint_name = 'linear'
    fcomp_name = 'fcnn'
    compute_selector(ds_name, fint_name, fcomp_name, 'logistic_regression', ps)
    compute_selector(ds_name, fint_name, fcomp_name, 'random_forest_128', ps)
    compute_selector(ds_name, fint_name, fcomp_name, 'random_selector', ps)
    compute_selector(ds_name, fint_name, fcomp_name, 'upsample_logistic_regression', ps)
    compute_selector(ds_name, fint_name, fcomp_name, 'upsample_tree', ps)
    compute_selector(ds_name, fint_name, fcomp_name, 'upsample_forest', ps)
    
    # Put results in table
    compute_selection_accuracy(ds_name, ps)

if __name__ == '__main__':
    main()