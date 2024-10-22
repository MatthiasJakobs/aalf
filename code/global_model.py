import tqdm
import pickle
import numpy as np
from gluonts.dataset.repository import get_dataset
from gluonts.dataset.util import to_pandas
from tsx.datasets import split_proportion, windowing
from models import CNN
from tsx.utils import string_to_randomstate
from os.path import exists
from os import makedirs
from config import Ls, Ns, tsx_to_gluon
from torchsummary import summary

H = 1
p_train = 0.7
makedirs('data', exist_ok=True)
makedirs('models', exist_ok=True)

for ds_name in ['nn5']:

    L = Ls[ds_name]
    n_datapoints = Ns[ds_name]

    if exists(f'data/{ds_name}.npz'):
        ds = np.load(f'data/{ds_name}.npz')
        x_train, y_train, x_val, y_val = ds['xtrain'], ds['ytrain'], ds['xval'], ds['yval']
    else:
        x_train, y_train = [], []
        x_val, y_val = [], []

        ds = get_dataset(tsx_to_gluon[ds_name])
        I = iter(ds.train)

        # Take p-percent as train and the rest as validation
        rng = string_to_randomstate(ds_name)

        indices = np.arange(n_datapoints)
        # Further subsample web traffic because of its size
        if ds_name == 'web_traffic':
            #indices = rng.choice(indices, size=3000, replace=False)
            indices = np.arange(3000)

        rng.shuffle(indices)
        cutoff = int(p_train * len(indices))
        train_indices = indices[:cutoff]
        val_indices = indices[cutoff:]

        # Collect data
        for i, _x in tqdm.tqdm(enumerate(I), desc=ds_name, total=len(indices)):
            X = to_pandas(_x).to_numpy().squeeze()
            X_train = X[:int(0.5*len(X))]

            mu = np.mean(X_train)
            std = np.std(X_train)
            if std <= 1e-6:
                continue
            X_train = (X_train - mu ) / std

            _x, _y = windowing(X_train, L=L, H=H)

            # Remove rows with nans in them
            to_keep = np.where(~np.any(np.isnan(_x), axis=1))[0]
            _x = _x[to_keep]
            _y = _y[to_keep]
            
            # If all entries are identical: remove
            to_keep = np.where(~np.all(_x[:, 0:1] == _x, axis=1))[0]
            _x = _x[to_keep]
            _y = _y[to_keep]

            if i in train_indices:
                x_train.append(_x)
                y_train.append(_y)
            elif i in val_indices:
                x_val.append(_x)
                y_val.append(_y)

            if i >= len(indices):
                break

        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        x_val = np.concatenate(x_val, axis=0)
        y_val = np.concatenate(y_val, axis=0)

        np.savez(f'data/{ds_name}.npz', xtrain=x_train, ytrain=y_train, xval=x_val, yval=y_val)

    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    # limit = 100000
    # x_train = x_train[:limit]
    # y_train = y_train[:limit]
    # x_val = x_val[:limit]
    # y_val = y_val[:limit]

    hyperparameters = {
        'nn5': {'batch_norm': True, 'n_hidden_channels': 64, 'depth_feature': 4, 'depth_classification': 2, 'max_epochs': 20, 'lr': 2e-5, 'batch_size': 128 },
        'pedestrian_counts': {'batch_norm': True, 'n_hidden_channels': 16, 'depth_feature':4, 'max_epochs': 50, 'lr': 1e-3, 'batch_size': 1024 },
        'weather': {'batch_norm': True, 'n_hidden_channels': 128, 'depth_feature': 3, 'n_hidden_neurons': 64, 'max_epochs': 10, 'lr': 2e-4, 'batch_size': 512 },
        'web_traffic': {'batch_norm': True, 'n_hidden_channels': 64, 'depth_feature': 4, 'max_epochs': 10, 'lr': 2e-4, 'batch_size': 512 },
        'kdd_cup_nomissing': {'batch_norm': True, 'n_hidden_channels': 16, 'depth_feature': 4, 'max_epochs': 10, 'lr': 2e-4, 'batch_size': 512 },
    }

    cnn = CNN(L, **hyperparameters[ds_name])
    cnn.fit(x_train, y_train, x_val=x_val, y_val=y_val)
    with open(f'models/cnn_{ds_name}.pickle', 'wb') as f:
        pickle.dump(cnn, f)