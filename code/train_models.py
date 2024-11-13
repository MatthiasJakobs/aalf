import pickle
import torch
import numpy as np
import skorch
from utils import rmse
from sklearn.linear_model import LinearRegression
from tsx.datasets import windowing
from sklearn.neural_network import MLPRegressor
from preprocessing import generate_covariates, load_global_data, load_local_data
from seedpy import fixedseed
from config import DATASET_HYPERPARAMETERS, DEEPAR_HYPERPARAMETERS, FCN_HYPERPARAMETERS
from os import makedirs
from tsx.utils import string_to_randomstate, get_device
from tsx.models import NeuralNetRegressor
from skorch.callbacks import EarlyStopping
from Trainer import GlobalTorchDataset
from models import DeepAR, FCNN
import torch.nn as nn

# def get_chronos_predictions(ds_name, n_samples=25, batch_size=128):
#     # Get data

#     # Instantiate model
#     device = get_device()
#     pipeline = ChronosPipeline.from_pretrained(
#         'amazon/chronos-t5-tiny',
#         device_map=device, 
#         torch_dtype=torch.bfloat16,
#     )
    
#     Xs, horizons, indices = load_dataset(ds_name, fraction=1)

#     L = 48
#     chronos_errors = []
#     linear_errors = []
#     H = 1
#     for ds_index in tqdm.tqdm(indices):
#         X = Xs[ds_index].to_numpy()

#         (x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocess_data(X, L, H)

#         lm = LinearRegression()
#         lm.fit(x_train, y_train)
#         preds = lm.predict(x_test).reshape(y_test.shape)
#         linear_errors.append(rmse(preds, y_test))

#         x_test = torch.from_numpy(x_test).float()

#         if x_test.shape[0] <= batch_size:
#             preds = pipeline.predict(context=x_test, prediction_length=H, num_samples=n_samples, top_k=45).mean(axis=1).numpy()
#         else:
#             preds = []
#             for i in tqdm.trange(0, len(x_test), batch_size):
#                 preds.append(pipeline.predict(context=x_test[i:i+batch_size], prediction_length=H, num_samples=n_samples, top_k=15).mean(axis=1).numpy())

#             preds = np.concatenate(preds, axis=0)

#         if H > 1:
#             preds = preds[:, -1:]
#         chronos_errors.append(rmse(preds, y_test))

#     print(f'chronos: {min(chronos_errors):.3f} {max(chronos_errors):.3f} {np.mean(chronos_errors):.3f}')
#     print(f'linear: {min(linear_errors):.3f} {max(linear_errors):.3f} {np.mean(linear_errors):.3f}')
#     print(np.argmax(chronos_errors))

#     fig, axs = plt.subplots(1, 2, figsize=(12, 3), sharey=True)
#     axs[0].bar(np.arange(len(linear_errors)), linear_errors)
#     axs[0].set_title('linear')
#     axs[1].bar(np.arange(len(chronos_errors)), chronos_errors)
#     axs[1].set_title('chronos')
#     fig.tight_layout()
#     fig.savefig('test.png')

#     fig, axs = plt.subplots(1, 1)
#     axs.scatter(linear_errors, chronos_errors)
#     axs.set_xlabel('linear error')
#     axs.set_ylabel('chronos error')
#     axs.plot([0, 1], [0, 1], '-', transform=axs.transAxes)
#     fig.tight_layout()
#     fig.savefig('scatter.png')

# def train_boosting(ds_name):
#     from catboost import CatBoostRegressor

#     Xs, horizons, indices = load_dataset(ds_name, fraction=1)

#     L = 10

#     errors = []
#     for ds_index in tqdm.tqdm(indices):
#         H = int(horizons[ds_index])
#         X = Xs[ds_index].to_numpy()

#         (x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocess_data(X, L, H)
#         model = CatBoostRegressor(iterations=200, depth=10, learning_rate=0.1, random_state=7)
#         model.fit(x_train, y_train, verbose=False)

#         preds = model.predict(x_test).reshape(y_test.shape)
#         errors.append(rmse(preds, y_test))

#     print(f'catboost: {min(errors):.3f} {max(errors):.3f} {np.mean(errors):.3f}')

def fit_fcnn(ds_name):
    random_state = string_to_randomstate(ds_name, return_seed=True)
    dsh = DATASET_HYPERPARAMETERS[ds_name]
    L = dsh['L']
    freq = dsh['freq']

    ds_train = GlobalTorchDataset(ds_name, freq, L, 1, split='train', return_X_y=True)

    # Get some data for validation
    rng = np.random.RandomState(random_state)
    val_indices = rng.binomial(n=1, p=0.1, size=len(ds_train))

    ds_val = torch.utils.data.Subset(ds_train, np.where(val_indices)[0])
    ds_train = torch.utils.data.Subset(ds_train, np.where(np.logical_not(val_indices))[0])

    makedirs(f'models/{ds_name}/', exist_ok=True)
    
    with fixedseed([torch, np], seed=random_state):
        mlp = FCNN(L, **FCN_HYPERPARAMETERS[ds_name])
        mlp.fit(ds_train, ds_val, verbose=True)

    (_, _), (_, _), (X_test, y_test) = load_global_data(ds_name, L=L, H=1, freq=freq)

    losses = []
    for _X_test, _y_test in zip(X_test, y_test):
        _X_test = _X_test.reshape(_X_test.shape[0], -1).astype(np.float32)
        test_preds = mlp.predict(_X_test).reshape(_y_test.shape)
        loss = rmse(test_preds, _y_test)
        losses.append(loss)
    print('fcnn', np.mean(losses))

    with open(f'models/{ds_name}/fcnn.pickle', 'wb') as _f:
        pickle.dump(mlp, _f)


def fit_deepar(ds_name):
    random_state = string_to_randomstate(ds_name, return_seed=True)
    dsh = DATASET_HYPERPARAMETERS[ds_name]
    L = dsh['L']
    freq = dsh['freq']

    ds_train = GlobalTorchDataset(ds_name, freq, L, 1, split='train', return_X_y=False)

    # Get some data for validation
    rng = np.random.RandomState(random_state)
    val_indices = rng.binomial(n=1, p=0.1, size=len(ds_train))

    ds_val = torch.utils.data.Subset(ds_train, np.where(val_indices)[0])
    ds_train = torch.utils.data.Subset(ds_train, np.where(np.logical_not(val_indices))[0])

    makedirs(f'models/{ds_name}/', exist_ok=True)

    with fixedseed([np, torch], seed=random_state):
        DAR = DeepAR(n_channel=4, **DEEPAR_HYPERPARAMETERS[ds_name])
        DAR.fit(ds_train, ds_val, verbose=True)

    (_, _), (_, _), (X_test, y_test) = load_global_data(ds_name, L=L, H=1, freq=freq)

    losses = []
    for _X_test, _y_test in zip(X_test, y_test):
        test_preds = DAR.predict(_X_test.astype(np.float32)).reshape(_y_test.shape)
        loss = rmse(test_preds, _y_test)
        losses.append(loss)
    print('deepar', np.mean(losses))

    with open(f'models/{ds_name}/deepar.pickle', 'wb') as _f:
        pickle.dump(DAR, _f)

def evaluate_models(ds_name):
    dsh = DATASET_HYPERPARAMETERS[ds_name]
    L = dsh['L']
    freq = dsh['freq']

    (local_X_train, local_y_train), (_, _), (local_X_test, local_y_test) = load_local_data(ds_name, L=L, H=1)
    (_, _), (_, _), (global_X_test, global_y_test) = load_global_data(ds_name, L=L, H=1, freq=freq)

    print('---')
    print(ds_name)

    for m_name in ['linear', 'fcnn', 'deepar']:

        if m_name == 'linear':
            losses = []
            for ds_index in range(len(local_X_train)):
                m = LinearRegression()
                m.fit(local_X_train[ds_index], local_y_train[ds_index])
                test_preds = m.predict(local_X_test[ds_index]).reshape(local_y_test[ds_index].shape)
                loss = rmse(test_preds, local_y_test[ds_index])
                losses.append(loss)
            print(f'{m_name}: {np.mean(losses):.3f}')
            continue
        elif m_name == 'deepar':
            with open(f'models/{ds_name}/deepar.pickle', 'rb') as f:
                m = pickle.load(f).to('cpu')
                m.device = 'cpu'
                m.lstm.flatten_parameters()
        elif m_name == 'fcnn':
            with open(f'models/{ds_name}/fcnn.pickle', 'rb') as f:
                m = pickle.load(f)
                m.device = 'cpu'
                m.model = m.model.to('cpu')
        else:
            raise NotImplementedError('Unknown model', m_name)
        losses = []
        for X_test, y_test in zip(global_X_test, global_y_test):
            if m_name == 'fcnn':
                X_test = X_test.reshape(X_test.shape[0], -1)
            test_preds = m.predict(X_test).reshape(y_test.shape)
            loss = rmse(test_preds, y_test)
            losses.append(loss)
        print(f'{m_name}: {np.mean(losses):.3f}')


def main():
    fit_deepar('weather')
    fit_deepar('nn5_daily_nomissing')
    fit_deepar('australian_electricity_demand')
    fit_deepar('pedestrian_counts')

    fit_fcnn('weather')
    fit_fcnn('nn5_daily_nomissing')
    fit_fcnn('australian_electricity_demand')
    fit_fcnn('pedestrian_counts')

    evaluate_models('weather')
    evaluate_models('nn5_daily_nomissing')
    evaluate_models('australian_electricity_demand')
    evaluate_models('pedestrian_counts')

if __name__ == '__main__':
    main()