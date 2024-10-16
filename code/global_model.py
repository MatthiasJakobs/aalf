import tqdm
import pickle
import numpy as np
from gluonts.dataset.repository import get_dataset
from gluonts.dataset.util import to_pandas
from tsx.datasets import split_proportion, windowing
from models import CNN
from tsx.utils import string_to_randomstate
from os.path import exists

dataset_names = ['weather', 'pedestrian_counts', 'kaggle_web_traffic_without_missing', 'kdd_cup_2018_without_missing']

Ls = {
    'weather': 30,
    'pedestrian_counts': 48,
    'kaggle_web_traffic_without_missing': 30,
    'kdd_cup_2018_without_missing': 48,
}

Ns = {
    'weather': 3010,
    'pedestrian_counts': 66,
    'kaggle_web_traffic_without_missing': 145063,
    'kdd_cup_2018_without_missing': 270,
}

H = 1
p_train = 0.7

for ds_name in ['pedestrian_counts']:

    L = Ls[ds_name]
    n_datapoints = Ns[ds_name]

    if exists(f'data/{ds_name}.npz'):
        ds = np.load(f'data/{ds_name}.npz')
        x_train, y_train, x_val, y_val = ds['xtrain'], ds['ytrain'], ds['xval'], ds['yval']
    else:
        x_train, y_train = [], []
        x_val, y_val = [], []

        ds = get_dataset(ds_name)
        I = iter(ds.train)

        # Take p-percent as train and the rest as validation
        rng = string_to_randomstate(ds_name)

        indices = np.arange(n_datapoints)
        # Further subsample web traffic because of its size
        if ds_name == 'kaggle_web_traffic_without_missing':
            indices = rng.choice(indices, size=3000, replace=False)

        rng.shuffle(indices)
        cutoff = int(p_train * n_datapoints)
        train_indices = indices[:cutoff]
        val_indices = indices[cutoff:]

        # Collect data
        for i, _x in tqdm.tqdm(enumerate(I), desc=ds_name, total=len(indices)):
            X = to_pandas(_x).to_numpy().squeeze()
            X_train = X[:int(0.5*len(X))]
            try:
                mu = np.mean(X_train)
                std = np.std(X_train)
            except RuntimeWarning:
                continue

            X_train = (X_train - mu ) / std

            _x, _y = windowing(X_train, L=L, H=H)

            if np.isnan(_x).any():
                continue
            if np.all(_x[:, 0:1] == _x, axis=1).any():
                continue

            if i in train_indices:
                x_train.append(_x)
                y_train.append(_y)
            elif i in val_indices:
                x_val.append(_x)
                y_val.append(_y)

        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        x_val = np.concatenate(x_val, axis=0)
        y_val = np.concatenate(y_val, axis=0)

        np.savez(f'data/{ds_name}.npz', xtrain=x_train, ytrain=y_train, xval=x_val, yval=y_val)

    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    # limit = 10000
    # x_train = x_train[:limit]
    # y_train = y_train[:limit]
    # x_val = x_val[:limit]
    # y_val = y_val[:limit]

    cnn = CNN(L, batch_norm=True, n_hidden_channels=16, depth_feature=4, max_epochs=20, lr=1e-3, batch_size=1024)
    cnn.fit(x_train, y_train, x_val=x_val, y_val=y_val)
    with open(f'models/cnn_{ds_name}.pickle', 'wb') as f:
        pickle.dump(cnn, f)