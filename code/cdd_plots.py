import numpy as np
import pandas as pd
from critdd import Diagram
from os import remove

TREATMENT_DICT = {
    'lin': r'$f_i$',
    'linear': r'$f_i$',
    'nn': r'$f_c$',
    'selOpt': 'Optimal Selection',
    'v12': 'AALF',
    'ade': 'ADE',
    'dets': 'DETS',
    'knnroc': 'KNN-RoC',
    'oms': 'OMS-RoC',
    'v12_0.5': r'AALF$_{0.5}$',
    'v12_0.6': r'AALF$_{0.6}$',
    'v12_0.7': r'AALF$_{0.7}$',
    'v12_0.8': r'AALF$_{0.8}$',
    'v12_0.9': r'AALF$_{0.9}$',
    'v12_0.95': r'AALF$_{0.95}$',
    'v12_0.99': r'AALF$_{0.99}$',
    'NewOracle50': r'Oracle$_{0.5}$',
    'NewOracle70': r'Oracle$_{0.7}$',
    'NewOracle90': r'Oracle$_{0.9}$',
}

DATASET_DICT = {
    'kdd_cup_nomissing': 'KDD Cup 2018',
    'weather': 'Weather',
    'pedestrian_counts': 'Pedestrian Counts',
    'london_smart_meters_nomissing': 'London Smart Meters',
    'web_traffic': 'Web Traffic'
}

DATASET_DICT_SMALL = {
    'kdd_cup_nomissing': 'KDD',
    'weather': 'Weather',
    'pedestrian_counts': 'Pedestrian',
    'london_smart_meters_nomissing': 'LSM',
    'web_traffic': 'Web'
}

def create_cdd_overall(treatment_names):
    ds_names = ['web_traffic', 'kdd_cup_nomissing', 'weather', 'pedestrian_counts']

    Xs = []

    for ds_name in ds_names:

        df_test = pd.read_csv(f'results/{ds_name}_test.csv')
        df_test = df_test.set_index('dataset_names')

        # Ensure same order of treatments
        df_test = df_test[treatment_names]
        Xs.append(df_test.to_numpy())

    treatment_names = [TREATMENT_DICT.get(key,key) for key in treatment_names]
    ds_names = [DATASET_DICT.get(key,key) for key in ds_names]

    print(np.vstack(Xs).shape)

    print(treatment_names)
    diagram = Diagram(
        np.vstack(Xs),
        treatment_names = treatment_names,
        maximize_outcome = False
    )

    diagram.to_file(
        f"total.pdf",
        axis_options = { # style the plot
            "width": "347.12354pt",
            "height": "80pt",
        },
        adjustment='bonferroni',
        alpha=0.05,
        escape_underscore=False,
    )

    # Cleanup temp files
    remove(f'total.aux')
    remove(f'total.log')

def main():
    print('create overall cdd')
    create_cdd_overall(treatment_names=['linear', 'nn', 'v12_0.9', 'v12_0.8', 'v12_0.7', 'v12_0.6', 'v12_0.95', 'v12_0.99', 'v12_0.5', 'oms', 'knnroc', 'ade', 'dets'])

if __name__ == '__main__':
    main()
