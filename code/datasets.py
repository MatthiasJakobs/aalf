import numpy as np
import pandas as pd

from tsx.datasets.monash import load_monash
from cdd_plots import DATASET_DICT
from config import tsx_to_gluon
from gluonts.dataset.util import to_pandas
from gluonts.dataset.repository import get_dataset
from tsx.utils import string_to_randomstate

def _load_from_gluon(ds_name):
    ds_name = tsx_to_gluon[ds_name]
    ds = get_dataset(ds_name)
    X = [to_pandas(_x).to_numpy() for _x in iter(ds.train)]

    return X

def load_dataset(ds_name, fraction=1):
    X = _load_from_gluon(ds_name)
    rng = string_to_randomstate(ds_name)

    if ds_name == 'web_traffic':
        X = np.vstack(X)
        # Find ones without missing data
        to_take = np.where((X==0).sum(axis=1) == 0)[0]
        # Subsample since this one is too large
        to_take = rng.choice(to_take, size=3000, replace=False)
        X = X[to_take].tolist()

    # Choose subset
    run_size = int(len(X)*fraction)
    indices = rng.choice(np.arange(len(X)), size=run_size, replace=False)
    
    # Remove datapoints that contain NaNs after preprocessing (for example, if all values are the same)
    if ds_name == 'weather':
        indices = [idx for idx in indices if idx not in [943, 568, 2221, 2054, 537, 1795, 1215, 891, 1191, 1639, 678, 379, 1048, 1938, 1264, 2010, 1308, 1450, 1961, 1475  ] ]
    if ds_name == 'kdd_cup_nomissing':
        indices = [idx for idx in indices if idx not in [248, 251, 249, 267, 247, 252, 262, 250, 205] ]

    X = [X[idx] for idx in indices]

    return X 

def get_dataset_statistics():
    ds_names = ['web_traffic', 'weather','kdd_cup_nomissing', 'pedestrian_counts']
    total = 0
    df = pd.DataFrame(columns=['Dataset name', 'Nr. Datapoints', 'Min. Length', 'Max. Length', 'Avg. Length'])
    df = df.set_index('Dataset name')
    for ds_name in ds_names:
        X = load_dataset(ds_name)
        n_datapoints = len(X)
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
