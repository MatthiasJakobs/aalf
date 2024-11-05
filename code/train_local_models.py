import pickle
import tqdm
from config import Ls, tsx_to_gluon, all_datasets
from gluonts.dataset.repository import get_dataset
from preprocessing import get_train_data
from sklearn.linear_model import LinearRegression
from tsx.datasets import windowing
from tsx.models.forecaster.baselines import TableForestRegressor
from os import makedirs
from os.path import join

def _run_fit(model_class, hyperparameters, X, L, H):
    model = model_class(**hyperparameters)
    x, y = windowing(X, L=L, H=H)

    if len(x.shape) != 2:
        x = x.reshape(1, -1)
    if len(y.shape) != 1:
        y = y.reshape(1, -1)

    model.fit(x, y)
    return model

def train_local_model(model_name, ds_name, H=1, verbose=True):
    dataset = get_dataset(tsx_to_gluon[ds_name])
    Xs, _, _ = get_train_data(dataset)
    L = Ls[ds_name]

    if model_name == 'linear':
        model_class = LinearRegression
        hyperparameters = {}
    elif model_name == 'trf-32':
        model_class = TableForestRegressor
        hyperparameters = {'n_estimators': 32, 'random_state': 9437117, 'n_jobs': -1, 'include_raw': True}
    elif model_name == 'cn':
    else:
        raise NotImplementedError('Local model', model_name, 'not implemented')

    models = [_run_fit(model_class, hyperparameters, X, L, H) for X in tqdm.tqdm(Xs, desc=f'{ds_name} - {model_name}', disable=not verbose)]

    path = f'models/local/{ds_name}/'
    makedirs(path, exist_ok=True)

    with open(join(path, model_name + '.pickle'), 'wb') as f:
        pickle.dump(models, f)

if __name__ == '__main__':
    for ds_name in all_datasets:
        train_local_model('trf-32', ds_name)
