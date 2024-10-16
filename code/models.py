import numpy as np
import pickle
import torch
import torch.nn as nn
import tqdm

import torch.utils
import torch.utils.data
from tsx.models import NeuralNetRegressor, TSValidSplit
from skorch.callbacks import EarlyStopping
from itertools import product
from copy import deepcopy
from seedpy import fixedseed

from tsx.models.forecaster.baselines import TableForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from os import makedirs
from os.path import exists

def fit_basemodels(ds_name, ds_index, x_train, y_train, random_state=None):

    models = {
        'rf_64_2': RandomForestRegressor(n_estimators=64, max_depth=2, random_state=random_state),
        'tfr_64_2': TableForestRegressor(n_estimators=64, max_depth=2, random_state=random_state),
        'rf_64_4': RandomForestRegressor(n_estimators=64, max_depth=4, random_state=random_state),
        'tfr_64_4': TableForestRegressor(n_estimators=64, max_depth=4, random_state=random_state),
        'svr': SVR(),
    }

    for m_name, m in models.items():
        path = f'models/{ds_name}/{ds_index}/{m_name}.pickle'
    
        if not exists(path):
            models[m_name] = models[m_name].fit(x_train, y_train.reshape(-1))
            with open(path, 'wb') as F:
                pickle.dump(m, F)
        else:
            with open(path, 'rb') as F:
                models[m_name] = pickle.load(F)

    return models

class HetEnsemble:

    def __init__(self, models, agg='mean'):
        self.models = models
        self.agg = agg

    def fit(self, x_train, y_train):
        return self

    def predict(self, x_test):
        preds = np.stack([m.predict(x_test) for m in self.models])
        if self.agg == 'mean':
            preds = preds.mean(axis=0)
        elif self.agg == 'median':
            preds = preds.median(axis=0)
        else:
            # TODO: Probably better check this in constructor already
            raise NotImplementedError('Unknwon aggregation strategy', self.agg)
        return preds

class ResidualBlock(nn.Module):

    def __init__(self, input_filters, output_filters, dilation=1, batch_norm=True):
        super(ResidualBlock, self).__init__()
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv1d(input_filters, output_filters, 3, bias=not batch_norm, dilation=dilation, padding='same')
        self.conv2 = nn.Conv1d(output_filters, output_filters, 3, bias=not batch_norm, dilation=dilation, padding='same')
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        if batch_norm:
            self.batchnorm_1 = nn.BatchNorm1d(output_filters)
            self.batchnorm_2 = nn.BatchNorm1d(output_filters)
        self.skip = nn.Conv1d(input_filters, output_filters, kernel_size=1)

    def forward(self, x):
        z = self.conv1(x)
        if self.batch_norm:
            z = self.batchnorm_1(z)
        z = self.relu1(z)
        z = self.conv2(z)
        if self.batch_norm:
            z = self.batchnorm_2(z)
        x = self.skip(x) + z
        return x

class CNN(nn.Module):

    def __init__(self, L=10, H=1, n_channels=1, depth_feature=2, depth_classification=2, n_hidden_neurons=32, batch_norm=True, max_epochs=10, random_state=None, n_hidden_channels=16, batch_size=64, lr=2e-3):
        super().__init__()

        self.L = L
        self.H = H
        self.max_epochs = max_epochs
        self.lr = lr
        self.random_state = random_state
        self.n_channels = n_channels
        self.n_hidden_channels = n_hidden_channels
        self.n_hidden_neurons = n_hidden_neurons
        self.depth_feature = depth_feature
        self.depth_classification = depth_classification
        self.batch_norm = batch_norm
        self.device = 'mps'
        self.batch_size = batch_size

    def build_model(self):
        with fixedseed(torch, self.random_state):
            feature_extractor = nn.ModuleList()
            forecaster = nn.ModuleList()

            for i in range(self.depth_feature):
                if i == 0:
                    feature_extractor.append(ResidualBlock(self.n_channels, self.n_hidden_channels, batch_norm=self.batch_norm, dilation=(2**i)))
                else:
                    feature_extractor.append(ResidualBlock(self.n_hidden_channels, self.n_hidden_channels, batch_norm=self.batch_norm, dilation=(2**i)))

            if self.depth_classification == 1:
                forecaster.append(nn.Linear(self.L * self.n_hidden_channels, self.H))
            else:
                for i in range(self.depth_classification):
                    if i == 0:
                        forecaster.append(nn.Linear(self.L * self.n_hidden_channels, self.n_hidden_neurons))
                        forecaster.append(nn.ReLU())
                    elif i == self.depth_classification-1:
                        forecaster.append(nn.Linear(self.n_hidden_neurons, self.H))
                    else:
                        forecaster.append(nn.Linear(self.n_hidden_neurons, self.n_hidden_neurons))
                        forecaster.append(nn.ReLU())

        self.model = nn.Sequential(*feature_extractor, nn.Flatten(), *forecaster)
        self.model.to(self.device)
        #self.model = torch.compile(self.model)

    def forward(self, x):
        lv = x[..., -1:]
        x = (x - lv)
        pred = self.model(x)
        pred = pred + lv.squeeze()
        return pred

    def predict(self, X):
        # Convert to torch tensors
        X_tensor = torch.Tensor(X).float()

        if len(X_tensor.shape) == 2:
            X_tensor = X_tensor.unsqueeze(1)

        self.model.eval()
        with torch.no_grad():
            out = self.model(X_tensor).squeeze()

        return out.numpy()


    def fit(self, X, y, x_val=None, y_val=None):
        self.build_model()
        #self.model = nn.Linear(self.L, 1).to(self.device)
        # Convert to torch tensors
        X_tensor = torch.Tensor(X).float().to(self.device)
        y_tensor = torch.Tensor(y).float().to(self.device)

        if len(X_tensor.shape) == 2:
            X_tensor = X_tensor.unsqueeze(1)

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        x_train = X_tensor
        y_train = y_tensor

        if self.device in ['cuda', 'mps']:
            pin_memory = True
        else:
            pin_memory = False

        if x_val is not None and y_val is not None:
            x_val = torch.Tensor(x_val).float().to(self.device)
            y_val = torch.Tensor(y_val).float().to(self.device)
            if len(x_val.shape) == 2:
                x_val = x_val.unsqueeze(1)

            ds_val = torch.utils.data.TensorDataset(x_val, y_val)
            dl_val = torch.utils.data.DataLoader(ds_val, pin_memory=pin_memory, drop_last=True, batch_size=self.batch_size)
            has_val = True
        else:
            has_val = False

        ds_train = torch.utils.data.TensorDataset(x_train, y_train)

        with fixedseed(torch, self.random_state):
            dl_train = torch.utils.data.DataLoader(ds_train, pin_memory=pin_memory, drop_last=True, batch_size=self.batch_size, shuffle=True)

            # Training loop
            for epoch in range(self.max_epochs):
                train_losses = []
                self.model.train()
                for b_x, b_y in tqdm.tqdm(dl_train, desc=f'Epoch {epoch}', total=len(dl_train)):
                    self.optimizer.zero_grad(set_to_none=True)
                    outputs = self.model(b_x).squeeze()
                    loss = self.loss_function(outputs, b_y)
                    loss.backward()
                    self.optimizer.step()
                    train_losses.append(loss.item())
                
                if has_val:
                    self.model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for b_x, b_y in dl_val:
                            outputs = self.model(b_x).squeeze()
                            loss = self.loss_function(outputs, b_y)
                            val_losses.append(loss.item())

                    print(f'Epoch {epoch} train loss {np.mean(train_losses):.3f} val loss {np.mean(val_losses):.3f}')
                else:
                    print(f'Epoch {epoch} train loss {np.mean(train_losses):.3f}')

        self.model = self.model.to('cpu')
        
    
class SimpleAutoML:

    def __init__(self, base_model, random_state=10984727):
        self.random_state = random_state
        self.base_model = base_model
        self.distributions = dict(
            lr=[1e-2],
            #lr=[1e-2, 1e-3, 5e-3, 5e-4],
            max_epochs=[1000],
        )

    def fit(self, X, y):
        self.is_fitted = False

        X = np.expand_dims(X.astype(np.float32), 1)
        y = y.astype(np.float32).reshape(-1, 1)

        # Get all parameter combinations
        values = self.distributions.values()
        keys = self.distributions.keys()
        all_params = [dict(zip(keys, combination)) for combination in product(*values)]

        # Fit all models
        all_models = []
        all_configs = []

        for params in all_params:
            m = NeuralNetRegressor(deepcopy(self.base_model), random_state=self.random_state, train_split=TSValidSplit(), verbose=True, device=None, callbacks=[EarlyStopping(load_best=True, patience=10)], **params)
            m.fit(X, y)
            best_val_score = m.callbacks[0].dynamic_threshold_
            did_stop_early = not (params['max_epochs'] == m.callbacks[0].best_epoch_)
            all_models.append(m)
            all_configs.append(params | {'best_score': best_val_score, 'did_stop_early': did_stop_early})

        # Get best
        best_model_index = np.argmin([c['best_score'] for c in all_configs])
        self.best_estimator_ = all_models[best_model_index]
        self.best_config_ = all_configs[best_model_index]

        self.is_fitted = True

    def __sklearn_is_fitted__(self):
        return self.is_fitted

    def predict(self, X):
        X = np.expand_dims(X.astype(np.float32), 1)
        return self.best_estimator_.predict(X)
        

class MedianPredictionEnsemble:

    def __init__(self, estimators):
        self.estimators = estimators

    @classmethod
    def load_model(cls, ds_name, ds_index, n=10):
        estimators = []
        for i in range(n):
            with open(f'models/{ds_name}/{ds_index}/nns/{i}.pickle', 'rb') as f:
                estimators.append(pickle.load(f))

        return cls(estimators)

    def save_model(self, ds_name, ds_index):
        makedirs(f'models/{ds_name}/{ds_index}/nns/', exist_ok=True)
        for idx, estimator in enumerate(self.estimators):
            with open(f'models/{ds_name}/{ds_index}/nns/{idx}.pickle', 'wb+') as f:
                pickle.dump(estimator, f)


    def fit(self, *args, **kwargs):
        for estimator in self.estimators:
            estimator.fit(*args, **kwargs)

    def predict(self, X):
        preds = []
        for estimator in self.estimators:
            preds.append(estimator.predict(X).reshape(-1, 1))
        
        return np.median(np.concatenate(preds, axis=-1), axis=-1).squeeze()

class UpsampleEnsembleClassifier:

    def __init__(self, model_class, n_member, *args, random_state=None, **kwargs):
        self.n_member = n_member
        self.rng = np.random.RandomState(random_state)
        self.estimators = [model_class(*args, random_state=self.rng, **kwargs) for _ in range(self.n_member)]

    def fit(self, X, y):
        one_indices = np.where(y == 1)[0]
        zero_indices = np.where(y == 0)[0]
        minority = int(np.mean(y) <= 0.5)

        # Upsample minority
        for i in range(self.n_member):
            # Upsample minority
            indices = self.rng.choice([zero_indices, one_indices][minority], size=len([zero_indices, one_indices][1-minority]), replace=True)
            indices = np.concatenate([indices, [zero_indices, one_indices][1-minority]])
            _x = X[indices]
            _y = y[indices]
            self.estimators[i].fit(_x, _y)

    def predict_proba(self, X):
        preds = np.concatenate([self.estimators[i].predict_proba(X)[:, 1].reshape(1, -1) for i in range(self.n_member)], axis=0)
        return preds.mean(axis=0), preds.std(axis=0)

    def predict(self, X, thresh=0.5):
        preds_proba, _ = self.predict_proba(X)
        return (preds_proba >= thresh).astype(np.int8)

    def global_feature_importance(self):
        return np.vstack([est.feature_importances_ for est in self.estimators]).mean(axis=0)
