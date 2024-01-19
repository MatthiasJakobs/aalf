import numpy as np
import pandas as pd

from tsx.datasets.monash import load_monash

def load_dataset(ds_name, fraction=1):
    if ds_name == 'web_traffic':
        X, horizons = load_monash('web_traffic_daily_nomissing', return_horizon=True)
        X = np.vstack([x.to_numpy() for x in X['series_value']])
        # Find ones without missing data
        to_take = np.where((X==0).sum(axis=1) == 0)[0]
        # Subsample since this one is too large
        to_take = np.random.RandomState(1234).choice(to_take, size=3000, replace=False)
        X = X[to_take]
        horizons = [horizons[i] for i in to_take]
    else:
        X, horizons = load_monash(ds_name, return_horizon=True)
        X = X['series_value']

    # Choose subset
    horizons = np.array(horizons)
    rng = np.random.RandomState(12389182)
    #run_size = len(X)
    run_size = int(len(X)*fraction)
    indices = rng.choice(np.arange(len(X)), size=run_size, replace=False)
    
    # Remove datapoints that contain NaNs after preprocessing (for example, if all values are the same)
    if ds_name == 'london_smart_meters_nomissing':
        indices = [idx for idx in indices if idx not in [ 65, 5531, 4642, 2846, 179, 2877, 5061, 920, 1440, 3076, 5538 ] ]
    if ds_name == 'weather':
        indices = [idx for idx in indices if idx not in [943] ]

    if ds_name not in [ 'weather', 'web_traffic' ]:
        horizons = np.ones((len(X))).astype(np.int8)

    return X, horizons, indices

def get_dataset_statistics():
    ds_names = ['web_traffic', 'london_smart_meters_nomissing', 'kdd_cup_nomissing', 'weather', 'pedestrian_counts']
    total = 0
    for ds_name in ds_names:
        df = pd.read_csv(f'results/{ds_name}_test.csv')
        n_datapoints = len(df)
        total += n_datapoints
        print(ds_name, n_datapoints)
    print('Total:', total)

if __name__ == '__main__':
    get_dataset_statistics()