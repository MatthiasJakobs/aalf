import tqdm
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from utils import rmse, load_dict_or_create, format_significant
from plotz import default_plot, COLORS
from config import DS_MAP, DATASET_HYPERPARAMETERS, ALL_DATASETS
from preprocessing import load_local_data, create_selector_features
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from os import makedirs
from models import RandomSelector, UpsampleEnsembleClassifier
from tsx.utils import string_to_randomstate

SELECTORS = {
    'random_selector': {
        'model_class': RandomSelector,
        'hyperparameters': {},
        'n_repeats': 100,
        'randomized': True,
        'name': r'$\mathtt{RND}$',
    },
    'logistic_regression': {
        'model_class': LogisticRegression,
        'hyperparameters': {'max_iter': 1000},
        'name': r'$\mathtt{LR}$',
    },
    'upsample_logistic_regression': {
        'model_class': UpsampleEnsembleClassifier,
        'hyperparameters': {'model_class': LogisticRegression, 'n_member': 9, 'random_state': 20241127, 'max_iter': 1000},
        'randomized': True,
        'name': r'$\mathtt{LRu}$',
    },
    'nn': {
        'model_class': MLPClassifier,
        'hyperparameters': {'early_stopping': True, 'max_iter': 500},
        'randomized': True,
        'name': r'$\mathtt{NN}$',
    },
    'upsample_nn': {
        'model_class': UpsampleEnsembleClassifier,
        'hyperparameters': {'model_class': MLPClassifier, 'n_member': 9, 'early_stopping': True, 'max_iter': 500},
        'randomized': True,
        'name': r'$\mathtt{NNu}$',
    },
    'random_forest_128': {
        'model_class': RandomForestClassifier,
        'hyperparameters': {'n_estimators': 128, 'random_state': 20241127},
        'randomized': True,
        'name': '$\mathtt{RF}$',
    },
    'upsample_forest': {
        'model_class': UpsampleEnsembleClassifier,
        'hyperparameters': {'model_class': RandomForestClassifier, 'n_member': 9, 'n_estimators': 128, 'random_state': 20241127},
        'randomized': True,
        'name': r'$\mathtt{RFu}$',
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
        #Bmax = (errors < self.threshold).sum() - 1
        Bmax = (errors < self.threshold).sum() 

        B = max(Bmax, B0)

        label = np.zeros((len(errors)))
        label[np.argsort(errors)[:B]] = 1

        return label.astype(np.int8)

def _run_selector(p, model_class, hyperparameters, n_repeats, X_train, y_train, X_val, y_val, X_test, y_test, fint_train_preds, fcomp_train_preds, fint_val_preds, fcomp_val_preds, fint_test_preds, fcomp_test_preds):

    # Get label
    oracle = Oracle(p)
    s_star_val = oracle.get_labels(y_val, fcomp_val_preds, fint_val_preds)
    s_star_test = oracle.get_labels(y_test, fcomp_test_preds, fint_test_preds)

    n_datapoints_with_label = np.unique(s_star_val, return_counts=True)[1]
    if n_datapoints_with_label[0] < 2 or n_datapoints_with_label[1] < 2:
        return 0, 0, 0, 0

    train_preds = np.vstack([fcomp_train_preds, fint_train_preds])
    val_preds = np.vstack([fcomp_val_preds, fint_val_preds])
    test_preds = np.vstack([fcomp_test_preds, fint_test_preds])
    _X_val = create_selector_features(X_train, y_train, X_val, y_val, train_preds, val_preds)
    _X_test = create_selector_features(X_val, y_val, X_test, y_test, val_preds, test_preds)

    model = model_class(**hyperparameters)
    tp, tn, fp, fn = 0, 0, 0, 0
    for _ in range(n_repeats):
        model.fit(_X_val, s_star_val)
        selector_preds = model.predict(_X_test).squeeze()

        _tn, _fp, _fn, _tp = confusion_matrix(s_star_test, selector_preds, labels=[0, 1]).ravel()
        tn += _tn
        fp += _fp
        fn += _fn
        tp += _tp

    return tp, tn, fp, fn

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
    name = SELECTORS[sel_name]['name']
    if 'randomized' in SELECTORS[sel_name].keys():
        hyperparameters['random_state'] = string_to_randomstate(f'{ds_name}_{sel_name}', return_seed=True)
    
    for p in ps:
        result = Parallel(n_jobs=-1, backend='loky')(delayed(_run_selector)(
            p, 
            model_class, 
            hyperparameters, 
            n_repeats, 
            local_X_train[ds_index], 
            local_y_train[ds_index], 
            local_X_val[ds_index], 
            local_y_val[ds_index], 
            local_X_test[ds_index], 
            local_y_test[ds_index], 
            preds['train'][fint_name][ds_index], 
            preds['train'][fcomp_name][ds_index], 
            preds['val'][fint_name][ds_index], 
            preds['val'][fcomp_name][ds_index], 
            preds['test'][fint_name][ds_index], 
            preds['test'][fcomp_name][ds_index]) for ds_index in tqdm.trange(n_datapoints, desc=f'[{ds_name} - {name} - {p}]')) 

        tp, tn, fp, fn = np.array(result).sum(axis=0)

        # Compute compound f1 score
        f1 = (2 * tp) / (2 * tp + fp + fn)

        try:
            results[sel_name]
        except KeyError:
            results[sel_name] = {}

        results[sel_name][p] = f1

    with open(f'results/selection/{ds_name}.pickle', 'wb') as f:
        pickle.dump(results, f)

def compute_selection_accuracy(ds_name, ps):
    with open(f'results/selection/{ds_name}.pickle', 'rb') as f:
        results = pickle.load(f)

    df = pd.DataFrame(columns=['modelname']+[str(p) for p in ps])
    model_names = list(results.keys())
    for model_name in model_names:
        scores = {str(k): [float(v)] for k, v in results[model_name].items()} | {'modelname': [SELECTORS[model_name]['name']]}
        df = pd.concat([df, pd.DataFrame(scores)], ignore_index=True)

    df = df.set_index('modelname')
    df = df.loc[[SELECTORS[model_name]['name'] for model_name in SELECTORS.keys()]]
    return df

        
def highlight_max_column(col):
    formatted_col = col.copy()
    sorted_indices = col.argsort()

    # Highlight highest in bold, second highest in underscore
    formatted_col.iloc[sorted_indices[-1]] = fr'\textbf{{{formatted_col.iloc[sorted_indices[-1]]}}}'
    formatted_col.iloc[sorted_indices[-2]] = fr'\underline{{{formatted_col.iloc[sorted_indices[-2]]}}}'

    return formatted_col

def create_selector_table(ps):
    # Create large table
    dfs = {DS_MAP[ds_name]: compute_selection_accuracy(ds_name, ps) for ds_name in ALL_DATASETS}
    dfs = {k: v.apply(format_significant, axis=0) for k, v in dfs.items()}
    dfs = {k: v.apply(highlight_max_column, axis=0) for k, v in dfs.items()}
    dfs = pd.concat(dfs)

    # Rename columns
    dfs.columns = [fr'$p={p}$' for p in dfs.columns]
    print(dfs.columns)

    # Export to LaTeX
    latex_table = dfs.to_latex(
        #multicolumn=True, 
        multirow=True, 
        float_format='%.3f', 
        index_names=False,
        column_format=r'llllllll',
    )

    # Replace clines with midrules
    latex_table = latex_table.replace(r'\cline{1-8}', r'\midrule')
    # Remove last midrule (since its unecessary)
    rows = latex_table.split('\n')
    latex_table = '\n'.join(rows[:-4] + rows[-3:])

    with open('plots/classifier_table.tex', 'w') as f:
        f.write(latex_table)
        
def main():
    ps = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    # for ds_name in ALL_DATASETS:
    #     dsh = DATASET_HYPERPARAMETERS[ds_name]
    #     if 'fint' not in dsh or 'fcomp' not in dsh:
    #         continue
        
    #     fint_name = dsh['fint']
    #     fcomp_name = dsh['fcomp']
    #     for sel_name in SELECTORS.keys():
    #         compute_selector(ds_name, fint_name, fcomp_name, sel_name, ps)

    create_selector_table(ps)

if __name__ == '__main__':
    main()