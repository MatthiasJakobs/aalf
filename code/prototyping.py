import numpy as np
import pandas as pd
import tqdm
import argparse

from joblib import Parallel, delayed

from main import run_experiment
from datasets import load_dataset
from cdd_plots import create_cdd

def run_on_subset(override=None):
    L = 10
    if override is None:
        override = []

    # Load data
    ds_names = ['pedestrian_counts', 'london_smart_meters_nomissing', 'web_traffic', 'kdd_cup_nomissing', 'weather' ]
    ds_fraction = 0.1
    rng = np.random.RandomState(20240103)

    all_results = []
    all_selection = []

    for ds_name in ds_names:
        X, horizons, indices = load_dataset(ds_name)

        # Get fraction
        indices = rng.choice(indices, size=int(len(indices)*ds_fraction), replace=False)

        test_results = pd.read_csv(f'results/{ds_name}_test.csv')
        test_results = test_results.set_index('dataset_names')
        test_results = test_results.T.to_dict()

        test_selection = pd.read_csv(f'results/{ds_name}_selection.csv')
        test_selection = test_selection.set_index('dataset_names')
        test_selection = test_selection.T.to_dict()

        log_test, log_selection = zip(*Parallel(n_jobs=-1, backend='loky')(delayed(run_experiment)(ds_name, ds_index, X[ds_index], L, horizons[ds_index], test_results[ds_index], test_selection[ds_index], to_run=override) for ds_index in tqdm.tqdm(indices, desc=ds_name)))
        for d in log_test:
            del d['dataset_names']
        for d in log_selection:
            del d['dataset_names']

        all_results.extend(log_test)
        all_selection.extend(log_selection)

    log_test = pd.DataFrame(list(all_results))
    log_test.index.rename('dataset_names', inplace=True)
    log_test.to_csv(f'results/all_small_test.csv')
    log_selection = pd.DataFrame(list(all_selection))
    log_selection.index.rename('dataset_names', inplace=True)
    log_selection.to_csv(f'results/all_small_selection.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rapid prototyping of new selection methods')
    parser.add_argument("--override", help='', nargs='+', default=[])
    args = vars(parser.parse_args())

    run_on_subset(override=args['override'])
    create_cdd('all_small')