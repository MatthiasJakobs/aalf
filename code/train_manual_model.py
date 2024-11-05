import numpy as np
import torch 
import tqdm
import torch.nn as nn
import torch.utils

from config import Ls, tsx_to_gluon, all_datasets
from preprocessing import get_dataset, get_train_data
from tsx.datasets import windowing
from copy import deepcopy
from os import makedirs
from evaluation import rmse

class Trainer:
    def __init__(self, model, learning_rate=2e-3, weight_decay=1e-6, limit_train_batches=None, device='cuda', patience=10, max_epochs=100):
        self.model = model.to(device)
        self.criterion = nn.MSELoss() 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.limit_train_batches = limit_train_batches
        self.device = device
        self.patience = patience
        self.max_epochs = max_epochs

    def fit(self, train_loader, dl_val):

        best_val_score = np.inf
        best_model = None
        since_improvement = 0
        
        for epoch in range(self.max_epochs):
            epoch_loss = []

            tl_length = len(train_loader)
            if self.limit_train_batches is not None:
                if self.limit_train_batches < 1:
                    limit = int(self.limit_train_batches * tl_length)
                else:
                    limit = self.limit_train_batches
            else:
                limit = tl_length
            
            self.model.train()
            for i, (bx, by) in tqdm.tqdm(enumerate(train_loader), total=limit, desc='train'):
                bx = bx.to(self.device)
                by = by.to(self.device)
                # Forward pass
                predictions = self.model(bx).squeeze()
                loss = self.criterion(predictions, by.squeeze())
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i >= limit:
                    break

                epoch_loss.append(loss.item())

            self.model.eval()
            with torch.no_grad():
                val_loss = []
                for i, (bx, by) in tqdm.tqdm(enumerate(dl_val), total=len(dl_val), desc='validation'):
                    bx = bx.to(self.device)
                    by = by.to(self.device)
                    # Forward pass
                    predictions = self.model(bx).squeeze()
                    loss = self.criterion(predictions, by.squeeze())
                    val_loss.append(loss.item())

                val_loss = np.mean(val_loss) 

            if val_loss <= best_val_score:
                since_improvement = 0
                best_val_score = val_loss
                best_model = deepcopy(self.model.state_dict())
            
                print(f'Epoch [{epoch+1}/{self.max_epochs}], train loss: {np.mean(epoch_loss):.3f} val loss: {val_loss:.3f} (new best val epoch)')
            else:
                since_improvement += 1
                print(f'Epoch [{epoch+1}/{self.max_epochs}], train loss: {np.mean(epoch_loss):.3f} val loss: {val_loss:.3f}')

            if since_improvement >= self.patience:
                print('End training and return to best model')
                break

        self.model.load_state_dict(best_model)

ds_hyperparameters = {
    'weather': { 'limit_train_batches': 0.1 },
    'tourism_monthly': { 'learning_rate': 1e-4 },
    'tourism_quarterly': { 'learning_rate': 1e-4 }
}

def train_manual_model(model_name, ds_name, H=1, max_epochs=100, early_stopping_patience=10):

    # Load training data or create it
    L = Ls[ds_name]

    dataset = get_dataset(tsx_to_gluon[ds_name])
    Xs, _, _ = get_train_data(dataset, gluon=False)

    p_val = 0.2

    all_X_train = []
    all_y_train = []
    all_X_val = []
    all_y_val = []
    for X in Xs:
        cutoff = int(len(X) * p_val)
        X_val = X[-cutoff:]
        X_train = X[:-cutoff]
        if len(X_train) <= L+H+1 or len(X_val) <= L+H+1:
            continue
        mu, std = np.mean(X_train), np.std(X_train)
        if np.abs(std) <= 1e-5:
            continue
        X_train = (X_train - mu) / std
        X_val = (X_val - mu) / std

        try:
            _X, _y = windowing(X_train, L=L, H=H)
            all_X_train.append(_X)
            all_y_train.append(_y)

            _X, _y = windowing(X_val, L=L, H=H)
            all_X_val.append(_X)
            all_y_val.append(_y)
        except RuntimeError:
            continue
    
    all_X_train = torch.from_numpy(np.concatenate(all_X_train, axis=0)).float()
    all_y_train = torch.from_numpy(np.concatenate(all_y_train, axis=0)).float()
    all_X_val = torch.from_numpy(np.concatenate(all_X_val, axis=0)).float()
    all_y_val = torch.from_numpy(np.concatenate(all_y_val, axis=0)).float()

    print(all_X_train.shape)
    print(all_y_train.shape)

    ds_train = torch.utils.data.TensorDataset(all_X_train, all_y_train)
    dl_train = torch.utils.data.DataLoader(ds_train, shuffle=True, batch_size=128)
    ds_val = torch.utils.data.TensorDataset(all_X_val, all_y_val)
    dl_val = torch.utils.data.DataLoader(ds_val, shuffle=False, batch_size=128)

    # Initialize models
    if model_name == 'fcn':
        model = nn.Sequential(
            nn.Linear(L, 100),
            nn.ReLU(),
            nn.Linear(100, H)
        )
    elif model_name == 'deepar':
        from deepar import DeepAR, generate_covariates
        model = DeepAR(n_channel=1, num_layers=1, dropout=0, device='cuda', batch_size=128, max_epochs=3, learning_rate=1e-2)
    else:
        raise NotImplementedError('Unknown global model', model_name)

    
    if model_name == 'deepar':
        model.fit(all_X_train.numpy()[..., None])
        val_preds = model.predict(all_X_val.numpy()[..., None], length=1, num_samples=10)
        val_preds = np.mean(val_preds, axis=-1).squeeze()

        print(rmse(val_preds, all_y_val.numpy().squeeze()))
    else:
        trainer = Trainer(model, **(ds_hyperparameters.get(ds_name, {})))
        trainer.fit(dl_train, dl_val)

        makedirs(f'models/global/{ds_name}', exist_ok=True)
        torch.save(trainer.model.state_dict(), f'models/global/{ds_name}/{model_name}.pth')

if __name__ == '__main__':
    for ds_name in ['tourism_monthly']:
        print('--------------')
        print(ds_name)
        print('--------------')
        train_manual_model('deepar', ds_name)
        