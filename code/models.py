import numpy as np
import tqdm
import torch
import torch.nn as nn
from tsx.utils import get_device, EarlyStopping
from preprocessing import _load_data, generate_covariates

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


class GlobalTorchDataset(torch.utils.data.Dataset):

    def __init__(self, ds_name, freq, L, H, split='train', return_X_y=False):
        super().__init__()
        self.L = L
        self.H = H
        
        # Get data
        X_train, X_val, X_test, start_dates = _load_data(ds_name, return_start_dates=True)
        if split == 'train':
            X = X_train
        elif split == 'val':
            X = X_val
        elif split == 'test':
            X = X_test
        else:
            raise NotImplementedError('Unknown split', split)

        X = [torch.from_numpy(x).float() for x in X]

        self.covariates = [torch.from_numpy(generate_covariates(len(X[i]), freq, start_dates[i])).float() for i in range(len(X))]
        self.n_windows = [len(x)-L-H for x in X]
        self.cum_length = np.cumsum(self.n_windows)
        self.X = X
        self.return_X_y = return_X_y

    def __len__(self):
        return sum(self.n_windows)

    def __getitem__(self, index):
        
        # Calculate which series to take
        for ds_index, l in enumerate(self.cum_length):
            if l >= index:
                break

        X = self.X[ds_index]
        covs = self.covariates[ds_index]

        # Calculate which window to take
        if ds_index == 0:
            remainder = index
        else:
            remainder = index - self.cum_length[ds_index-1]

        # win_i : [i:L+H+i]
        X_window = X[remainder:(remainder+self.L+self.H)].reshape(-1, 1)
        cov_window = covs[remainder:(remainder+self.L+self.H)]

        if self.return_X_y:
            # Construct window
            x = torch.cat([X_window[:self.L], cov_window[:self.L]], axis=-1)
            y = X_window[self.L:self.L+self.H]

            return x, y
        else:
            return torch.cat([X_window, cov_window], axis=-1)
    

class TorchBase(nn.Module):
    def __init__(self, learning_rate=2e-3, max_epochs=100, limit_train_batches=None, batch_size=64, device=None, patience=10, show_progress=True):
        super().__init__()
        self.device = get_device() if device is None else device
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.limit_train_batches = limit_train_batches
        self.patience = patience
        self.show_progress = show_progress

    @torch.no_grad()
    def predict(self, x):
        pass

    def train_epoch(self, e, dl, n_batches_per_epoch):
        pass
    
    @torch.no_grad
    def evaluate(self, e, dl):
        pass

    def fit(self, ds_train, ds_val, verbose=False):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        ES = EarlyStopping(patience=self.patience)

        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=self.batch_size, shuffle=True, num_workers=16)
        dl_val = torch.utils.data.DataLoader(ds_val, batch_size=self.batch_size, shuffle=False, num_workers=16)

        if self.limit_train_batches is not None:
            n_batches_per_epoch = min(self.limit_train_batches, len(dl_train))
        else:
            n_batches_per_epoch = len(dl_train)

        for epoch in range(self.max_epochs):
            epoch_loss = self.train_epoch(epoch, dl_train, n_batches_per_epoch)
            val_loss = self.evaluate(epoch, dl_val)

            ES.update(val_loss)
            if ES.best_score == val_loss:
                postfix = '(best epoch)'
            else:
                postfix = ''

            if verbose:
                print(f'Epoch [{epoch+1}/{self.max_epochs}], Train Loss: {np.mean(epoch_loss):.4f} Val Loss: {val_loss:.4f} {postfix}')

            if ES.should_stop():
                if verbose:
                    print('Stopped')
                break

class DeepAR(TorchBase):
    def __init__(self, n_channel=1, hidden_size=50, num_layers=2, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(n_channel, hidden_size, num_layers, batch_first=True, dropout=dropout).to(self.device)
        self.mean_estimator = nn.Linear(hidden_size*num_layers, 1).to(self.device)
        self.var_estimator = nn.Linear(hidden_size*num_layers, 1).to(self.device)

        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: 'bias' in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

    def initialize_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return h0, c0

    def forward(self, x, h0, c0):

        _, (h0, c0) = self.lstm(x, (h0, c0))

        mu = self.mean_estimator(h0.reshape(x.shape[0], -1)).squeeze()
        sigma = torch.log(1 + torch.exp(self.var_estimator(h0.reshape(x.shape[0], -1)))).squeeze()
        return mu, sigma, h0, c0

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        x = torch.from_numpy(x).float().to(self.device)
        batch_size = x.shape[0]
        input_size = x.shape[1]
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        # First, we get h0 and c0
        h0, c0 = self.initialize_hidden(batch_size)
        for t in range(input_size-1):
            _, _, h0, c0 = self(x[:, t].unsqueeze(1), h0, c0)

        mu, _, _, _ = self(x[:, -1].unsqueeze(1), h0, c0)
        return mu.cpu().numpy()

    def train_epoch(self, e, dl, n_batches_per_epoch):
        self.train()
        epoch_loss = 0
        n = 0
        for b_x in tqdm.tqdm(dl, desc=f'epoch {e} train', total=n_batches_per_epoch, disable=not self.show_progress):
            self.optimizer.zero_grad()
            b_x = b_x.to(self.device)

            h0, c0 = self.initialize_hidden(b_x.shape[0])
            loss = 0

            L = b_x.shape[1]

            for t in range(L-1):
                mu, sigma, h0, c0 = self(b_x[:, t].unsqueeze(1).clone(), h0, c0)
                ll = torch.distributions.normal.Normal(mu, sigma).log_prob(b_x[:, t+1, 0]).mean()
                loss += ll
            
            loss = -loss
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item() / L

            if n >= n_batches_per_epoch:
                break
            n += 1

        return epoch_loss / n_batches_per_epoch
    
    @torch.no_grad
    def evaluate(self, e, dl):
        self.eval()
        squared_errors = 0
        for b_x in tqdm.tqdm(dl, desc=f'epoch {e} evaluate', total=len(dl), disable=not self.show_progress):
            b_x = b_x.to(self.device)

            batch_size = b_x.shape[0]
            L = b_x.shape[1]

            # First, we get h0 and c0
            h0, c0 = self.initialize_hidden(batch_size)
            for t in range(L-1):
                mu, _, h0, c0 = self(b_x[:, t].unsqueeze(1), h0, c0)

            squared_errors += ((mu-b_x[:, -1, 0])**2).sum()

        return np.sqrt(squared_errors.cpu().numpy() / len(dl.dataset))

class FCNN(TorchBase):

    def __init__(self, L, hidden_size=20, **kwargs):
        super().__init__(**kwargs)
        self.model = nn.Sequential(
            nn.Linear(L * 4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(self.device)

    def train_epoch(self, e, dl, n_batches_per_epoch):
        criterion = nn.MSELoss()
        epoch_loss = 0.0
        n = 0
        for batch_X, batch_y in tqdm.tqdm(dl, total=n_batches_per_epoch, desc=f'epoch {e} train', disable=(not self.show_progress)):
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            batch_size = batch_X.shape[0]
            batch_X = batch_X.reshape(batch_size, -1)

            predictions = self.model(batch_X)
            loss = criterion(predictions.reshape(batch_y.shape), batch_y)
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()

            if n >= n_batches_per_epoch:
                break
            n += 1 

        return epoch_loss / n_batches_per_epoch

    @torch.no_grad
    def evaluate(self, e, dl):
        loss = 0.0
        for batch_X, batch_y in tqdm.tqdm(dl, total=len(dl), desc=f'epoch {e} evaluation', disable=(not self.show_progress)):
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            batch_size = batch_X.shape[0]
            batch_X = batch_X.reshape(batch_size, -1)

            predictions = self.model(batch_X).reshape(batch_y.shape)

            loss += ((batch_y - predictions)**2).sum()

        return np.sqrt((loss / len(dl.dataset)).cpu().numpy())

    @torch.no_grad
    def predict(self, X):
        # Ensure the model is in evaluation mode
        self.model.eval()
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        predictions = self.model(X)
        
        # Convert output to numpy for consistency with scikit-learn's output format
        return predictions.cpu().numpy()