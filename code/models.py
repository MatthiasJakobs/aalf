import numpy as np
import pickle
import torch
import torch.nn as nn

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

class HetEnsemble:

    def __init__(self, random_state=None, agg='mean'):
        self.random_state = random_state
        self.agg = agg

    def fit(self, x_train, y_train):
        self.models = [
            RandomForestRegressor(n_estimators=64, random_state=self.random_state),
            TableForestRegressor(n_estimators=64, random_state=self.random_state),
            SVR(),
        ]
        self.models = [m.fit(x_train, y_train.reshape(-1)) for m in self.models]

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

# Inspired by residual block found in ResNets
class ResidualBlock(nn.Module):

    def __init__(self, L, input_filters, output_filters, batch_norm=False, skip=True, dilation=0):
        super(ResidualBlock, self).__init__()

        self.batch_norm = batch_norm

        self.conv1 = nn.Conv1d(input_filters, output_filters, 3, dilation=2**dilation, padding='same')
        self.conv2 = nn.Conv1d(output_filters, output_filters, 3, dilation=2**dilation, padding='same')
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.skip_weights = nn.Conv1d(input_filters, output_filters, 1, padding='same')
        self.skip = skip
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(output_filters)
            self.bn2 = nn.BatchNorm1d(output_filters)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        z = self.conv1(x)
        if self.batch_norm:
            z = self.bn1(z)
        z = self.relu1(z)
        # DO
        #z = self.dropout(z)
        z = self.conv2(z)
        if self.batch_norm:
            z = self.bn2(z)
        z = self.relu2(z)
        if self.skip:
            return self.skip_weights(x) + z
        return z

class CNN(nn.Module):

    def __init__(self, L, n_filters=32, depth_classification=1, batch_norm=False, n_hidden_neurons=16, n_blocks=1, skip_connections=True, fit_normalized='none', max_epochs=10, lr=2e-3, random_state=None):
        super().__init__()
        self.fit_normalized = fit_normalized
        self.n_filters = n_filters
        self.L = L
        self.max_epochs = max_epochs
        self.lr = lr
        self.device = 'mps'
        self.random_state = random_state
        self.depth_classification = depth_classification
        self.feature_extractor = nn.Sequential(
            *[ResidualBlock(L, 1, n_filters, skip=skip_connections, dilation=i, batch_norm=batch_norm) if i == 0 else ResidualBlock(L, n_filters, n_filters, skip=skip_connections, dilation=i, batch_norm=batch_norm) for i in range(n_blocks)]
        )
        self.forecaster = [nn.Flatten()]
        if depth_classification == 1:
            self.forecaster.append(nn.Linear(L * n_filters, 1))
        else:
            for i in range(depth_classification):
                if i == 0:
                    self.forecaster.append(nn.Linear(L * n_filters, n_hidden_neurons))
                    self.forecaster.append(nn.ReLU())
                elif i == depth_classification-1:
                    self.forecaster.append(nn.Linear(n_hidden_neurons, 1))
                else:
                    self.forecaster.append(nn.Linear(n_hidden_neurons, n_hidden_neurons))
                    self.forecaster.append(nn.ReLU())
        self.forecaster = nn.Sequential(*self.forecaster)
        self.model = nn.Sequential(
            self.feature_extractor,
            self.forecaster
        )

        self.model = nn.Sequential(
            nn.Conv1d(1, self.n_filters, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(self.n_filters, self.n_filters, kernel_size=3, padding='same'),
            nn.Flatten(),
            nn.Linear(L * self.n_filters, 1)
        )

        self.model.to(self.device)

    def forward(self, X):
        if self.fit_normalized == 'mean':
            mu = X.mean(axis=-1, keepdims=True)
            _X = X-mu
            pred = self.model(_X)
            pred = (pred.unsqueeze(1) + mu).squeeze(1)
        elif self.fit_normalized == 'last':
            lv = X[..., -1:]
            _X = X-lv
            pred = self.model(_X)
            pred = (pred.unsqueeze(1) + lv).squeeze(1)
        else:
            pred = self.model(X)

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


    def fit(self, X, y, val_percent=0.3):
        # Convert to torch tensors
        X_tensor = torch.Tensor(X).float().to(self.device)
        y_tensor = torch.Tensor(y).float().to(self.device)

        if len(X_tensor.shape) == 2:
            X_tensor = X_tensor.unsqueeze(1)

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        train_size = int((1-val_percent) * X_tensor.shape[0])
        # x_train = X_tensor[:train_size]
        # y_train = y_tensor[:train_size]
        x_train = X_tensor
        y_train = y_tensor
        # x_val = X_tensor[train_size:]
        # y_val = y_tensor[train_size:]

        ds_train = torch.utils.data.TensorDataset(x_train, y_train)

        with fixedseed(torch, self.random_state):
            dl_train = torch.utils.data.DataLoader(ds_train, drop_last=True, batch_size=64, shuffle=True)

            # Training loop
            for epoch in range(self.max_epochs):
                train_losses = []
                self.model.train()
                for b_x, b_y in dl_train:
                    self.optimizer.zero_grad()
                    outputs = self.model(b_x).squeeze()
                    loss = self.loss_function(outputs, b_y)
                    loss.backward()
                    self.optimizer.step()
                    train_losses.append(loss.item())
                
                # self.model.eval()
                # with torch.no_grad():
                #     outputs = self.model(x_val).squeeze()
                #     val_loss = self.loss_function(outputs, y_val)

                #print(f'Epoch {epoch} train loss {np.mean(train_losses):.3f} val loss {val_loss.item():.3f}')
                #print(f'Epoch {epoch} train loss {np.mean(train_losses):.3f}')

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
