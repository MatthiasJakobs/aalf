import pandas as pd
from os import listdir, remove

DATASET_NAMES = ['kdd_cup_nomissing', 'pedestrian_counts', 'weather', 'web_traffic', 'london_smart_meters_nomissing']

def remove_from_results(methods):
    for ds_name in DATASET_NAMES:
        for extension in ['test', 'selection']:
            path = f'results/{ds_name}_{extension}.csv'
            try:
                res = pd.read_csv(path)
            except FileNotFoundError:
                continue

            res.drop(columns=methods, errors='ignore', inplace=True)
            res.set_index('dataset_names', inplace=True)
            res.to_csv(path)

def remove_from_preds(methods):
    for ds_name in DATASET_NAMES:
        indices = listdir(f'preds/{ds_name}/')
        for ds_index in indices:
            for method in methods:
                path = f'preds/{ds_name}/{ds_index}/{method}.npy'
                try:
                    remove(path)
                except FileNotFoundError:
                    continue

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--methods", help='', nargs='+', default=[])
    args = vars(parser.parse_args())
    methods = args['methods']

    remove_from_results(methods)
    remove_from_preds(methods)