import numpy as np
import pandas as pd

from tsx.datasets.monash import load_monash
from cdd_plots import DATASET_DICT

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
        indices = [idx for idx in indices if idx not in [943, 568, 2221, 2054, 537, 1795, 1215, 891, 1191, 1639, 678, 379, 1048, 1938, 1264, 2010, 1308, 1450, 1961, 1475  ] ]
    if ds_name == 'kdd_cup_nomissing':
        indices = [idx for idx in indices if idx not in [248, 251, 249, 267, 247, 252, 262, 250] ]

    if ds_name not in [ 'weather', 'web_traffic' ]:
        horizons = np.ones((len(X))).astype(np.int8)

    return X, horizons, indices

def get_dataset_statistics():
    ds_names = ['web_traffic', 'weather','kdd_cup_nomissing', 'pedestrian_counts']
    total = 0
    df = pd.DataFrame(columns=['Dataset name', 'Nr. Datapoints', 'Min. Length', 'Max. Length', 'Avg. Length'])
    df = df.set_index('Dataset name')
    for ds_name in ds_names:
        X, _, indices = load_dataset(ds_name)
        n_datapoints = len(indices)
        total += n_datapoints
        lengths = [len(x) for x in X]
        df.loc[DATASET_DICT.get(ds_name, ds_name)] = (int(n_datapoints), int(min(lengths)), int(max(lengths)), np.mean(lengths))
    df.loc['\\textbf{Total}'] = [total, np.nan, np.nan, np.nan]
    df['Nr. Datapoints'] = pd.to_numeric(df['Nr. Datapoints'], errors='coerce').astype('Int32')
    df['Min. Length'] = pd.to_numeric(df['Min. Length'], errors='coerce').astype('Int32')
    df['Max. Length'] = pd.to_numeric(df['Max. Length'], errors='coerce').astype('Int32')

    df = df.reset_index()
    tex = df.to_latex(na_rep='', float_format='%.2f', index=False)
    tex_list = tex.splitlines()
    tex_list.insert(len(tex_list)-3, '\midrule')

    # Use tabular* for correct width
    tex_list = ['\\begin{tabular*}{\\linewidth}{@{\extracolsep{\\fill}} lrrrr}'] + tex_list[1:-1] + ['\\end{tabular*}']
    tex = '\n'.join(tex_list)

    with open('plots/ds_table.tex', 'w+') as _f:
        _f.write(tex)


if __name__ == '__main__':
    get_dataset_statistics()
