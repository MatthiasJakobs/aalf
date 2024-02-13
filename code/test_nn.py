import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset 
from torch.optim.lr_scheduler import StepLR
from seedpy import fixedseed
from tsx.utils import get_device
from os.path import exists

import tqdm
import pickle
from os import listdir
from main import rmse
import pandas as pd
from cdd_plots import create_cdd

def load_or_compute(path, func, *args, **kwargs):
    if not exists(path):
        return func(*args, **kwargs)
    
    # Try torch
    try:
        return torch.load(path)
    except pickle.UnpicklingError:
        pass

    # Try numpy
    try:
        return np.load(path)
    except ValueError:
        pass

    raise NotImplementedError(f'Cannot load file {path}')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        x = self.fc3(x)
        output = F.sigmoid(x)
        return output

    def predict(self, x):
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        with torch.no_grad():
            return (self.forward(x) > 0.5).int().squeeze().numpy()


def train(epoch, model, device, train_loader, optimizer, _lambda):
    model.train()
    model.to(device)
    epoch_prediction_loss = []
    epoch_total_loss = []
    epoch_purity = []
    for (X, y) in (tbar := tqdm.tqdm(train_loader)):
        tbar.set_description(f'epoch {epoch} train')
        X, y = X.to(device), y.to(device)

        # Individual predictions
        fi_preds = X[:, 0]
        fc_preds = X[:, 1]

        optimizer.zero_grad()
        p_t = model(X).squeeze()
        prediction_loss = ((y - (p_t * fi_preds + (1-p_t) * fc_preds))**2)
        constrained_loss = (_lambda * (1-p_t))
        loss = (prediction_loss + constrained_loss).sum()
        loss.backward()
        optimizer.step()

        # Total loss
        epoch_total_loss.append(prediction_loss.detach() + constrained_loss.detach())

        # Quantized prediction loss
        p_t = (p_t > 0.5).float()
        prediction_loss = ((y - (p_t * fi_preds + (1-p_t) * fc_preds))**2)
        epoch_prediction_loss.append(prediction_loss)

        # Purity
        epoch_purity.append(p_t)

    return torch.mean(torch.hstack(epoch_total_loss)).item(), torch.mean(torch.hstack(epoch_prediction_loss)).item(), torch.mean(torch.hstack(epoch_purity)).item()


def test(epoch, model, device, test_loader, _lambda):
    model.eval()
    epoch_total_loss = []
    epoch_prediction_loss = []
    epoch_purity = []
    with torch.no_grad():
        for (X, y) in (tbar := tqdm.tqdm(test_loader)):
            tbar.set_description(f'epoch {epoch} val')
            X, y = X.to(device), y.to(device)

            # Individual predictions
            fi_preds = X[:, 0]
            fc_preds = X[:, 1]

            p_t = (model(X).squeeze() > 0.5).float()
            prediction_loss = ((y - (p_t * fi_preds + (1-p_t) * fc_preds))**2)
            constrained_loss = (_lambda * (1-p_t))

            # Total loss
            epoch_total_loss.append(prediction_loss.detach() + constrained_loss.detach())

            # Quantized prediction loss
            epoch_prediction_loss.append(prediction_loss)

            # Purity
            epoch_purity.append(p_t)

    return torch.mean(torch.hstack(epoch_total_loss)).item(), torch.mean(torch.hstack(epoch_prediction_loss)).item(), torch.mean(torch.hstack(epoch_purity)).item()

def train_global_model(train_dl, val_dl):
    device = get_device()
    n_epochs = 50
    #_lambda = 0
    _lambda = 0.05

    with fixedseed(torch, 102391):
        model = Net().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=4e-3)

        for epoch in range(n_epochs):
            train_total, train_prediction, train_purity = train(epoch, model, device, train_dl, optimizer, _lambda)
            val_total, val_prediction, val_purity = test(epoch, model, device, val_dl, _lambda)
            print(f'{train_total:.4f}, {train_prediction:.4f}, {train_purity:.2f} | {val_total:.4f}, {val_prediction:.4f} {val_purity:.2f}')
            torch.save(model.to('cpu').state_dict(), 'model.net')
            print('---'*30)

    return model.to('cpu').state_dict()

    

def global_model():

    rng = np.random.RandomState(92848372)
    ds_indices = listdir('data/optim_london_smart_meters_nomissing')

    log_test = []
    log_selection = []
    train_data = []
    val_data = []
    test_data = []
    for ds_index in tqdm.tqdm(ds_indices):
        if rng.binomial(1, p=0.95, size=1):
            continue
        save_x = np.load(f'data/optim_london_smart_meters_nomissing/{ds_index}/train_X.npy')
        save_y = np.load(f'data/optim_london_smart_meters_nomissing/{ds_index}/train_y.npy')
        X = torch.from_numpy(save_x).float()
        y = torch.from_numpy(save_y).float()
        ds = TensorDataset(X, y)
        
        if rng.binomial(1, p=0.7, size=1):
            train_data.append(ds)
        else:
            val_data.append(ds)

    train_ds = ConcatDataset(train_data)
    val_ds = ConcatDataset(val_data)
    print('Train size', len(train_ds))
    print('Val size', len(val_ds))
    batch_size = 4086
    train_dl = DataLoader(train_ds, batch_size=batch_size, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, pin_memory=True)

    weights = load_or_compute('model.net', train_global_model, train_dl, val_dl)
    model = Net()
    model.load_state_dict(weights)

    #print('selection percentage', np.mean(selection))
    for ds_index in tqdm.tqdm(ds_indices, desc='test'):
        test_results = {}
        selection_results = {}

        # Load test data
        x_test = np.load(f'data/optim_london_smart_meters_nomissing/{ds_index}/test_X.npy')
        y_test = np.load(f'data/optim_london_smart_meters_nomissing/{ds_index}/test_y.npy')

        #selection = model.predict(x_test)

        lin_preds_test = x_test[:, 0]
        nn_preds_test = x_test[:, 1]
        test_prediction_test = np.choose(np.ones(len(lin_preds_test)).astype(np.int8), [nn_preds_test, lin_preds_test])
        loss_test_test = rmse(test_prediction_test, y_test)
        #print('linear', loss_test_test)
        test_results['linear'] = loss_test_test

        test_prediction_test = np.choose(np.zeros(len(lin_preds_test)).astype(np.int8), [nn_preds_test, lin_preds_test])
        loss_test_test = rmse(test_prediction_test, y_test)
        #print('nn', loss_test_test)
        test_results['nn'] = loss_test_test

        # test_prediction_test = np.choose(selection, [nn_preds_test, lin_preds_test])
        # loss_test_test = rmse(test_prediction_test, y_test)
        # #print('new method', loss_test_test)
        # test_results['new_method'] = loss_test_test
        # selection_results['new_method'] = np.mean(selection)

    log_test.append(test_results)
    log_selection.append(selection_results)

    log_test = pd.DataFrame(list(log_test))
    log_test.index.rename('dataset_names', inplace=True)
    log_test.to_csv('results/NNlondon_test.csv')
    log_selection = pd.DataFrame(list(log_selection))
    log_selection.index.rename('dataset_names', inplace=True)
    log_selection.to_csv('results/NNlondon_selection.csv')

    create_cdd('NNlondon')

def main():

    ds_indices = listdir('data/optim_weather')

    log_test = []
    log_selection = []
    for ds_index in tqdm.tqdm(ds_indices):
        test_results = {}
        selection_results = {}
        #print('---'*10, ds_index, '---'*10)
        save_x = np.load(f'data/optim_weather/{ds_index}/train_X.npy')
        save_y = np.load(f'data/optim_weather/{ds_index}/train_y.npy')
        X = torch.from_numpy(save_x).float()
        y = torch.from_numpy(save_y).float()

        # Load test data
        x_test = np.load(f'data/optim_weather/{ds_index}/test_X.npy')
        y_test = np.load(f'data/optim_weather/{ds_index}/test_y.npy')
        
        # TODO: Split for validation
        train_ds = TensorDataset(X, y)
        train_dl = DataLoader(train_ds, batch_size=512)

        n_epochs = 1000
        with fixedseed(torch, 102391):
            model = Net()
            optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
            _lambda = 0.01

            for epoch in range(n_epochs):
                total_loss, prediction_loss, constrained_loss = train(model, 'cpu', train_dl, optimizer, _lambda)
                # val_loss, val_percent_ones = test(model, 'cpu', train_dl, _lambda)
                # if (epoch+1) % 100 == 0:
                #     print(epoch, total_loss, prediction_loss, constrained_loss, val_loss, val_percent_ones)

        selection = model.predict(x_test)
        #print('selection percentage', np.mean(selection))

        lin_preds_test = x_test[:, 0]
        nn_preds_test = x_test[:, 1]
        test_prediction_test = np.choose(np.ones(len(lin_preds_test)).astype(np.int8), [nn_preds_test, lin_preds_test])
        loss_test_test = rmse(test_prediction_test, y_test)
        #print('linear', loss_test_test)
        test_results['linear'] = loss_test_test

        test_prediction_test = np.choose(np.zeros(len(lin_preds_test)).astype(np.int8), [nn_preds_test, lin_preds_test])
        loss_test_test = rmse(test_prediction_test, y_test)
        #print('nn', loss_test_test)
        test_results['nn'] = loss_test_test

        test_prediction_test = np.choose(selection, [nn_preds_test, lin_preds_test])
        loss_test_test = rmse(test_prediction_test, y_test)
        #print('new method', loss_test_test)
        test_results['new_method'] = loss_test_test
        selection_results['new_method'] = np.mean(selection)

        log_test.append(test_results)
        log_selection.append(selection_results)

    log_test = pd.DataFrame(list(log_test))
    log_test.index.rename('dataset_names', inplace=True)
    log_test.to_csv('results/NNweather_test.csv')
    log_selection = pd.DataFrame(list(log_selection))
    log_selection.index.rename('dataset_names', inplace=True)
    log_selection.to_csv('results/NNweather_selection.csv')

    create_cdd('NNweather')

    

if __name__ == '__main__':
    #main()

    global_model()
    exit()
    import matplotlib.pyplot as plt
    df = pd.read_csv(f'results/NNweather_selection.csv')
    names = df.columns[1:]
    plt.figure()
    plt.violinplot(df.iloc[:, 1:].to_numpy(), showmedians=True)
    #plt.boxplot(df.iloc[:, 1:].to_numpy())
    epsilon = 0.05
    plt.ylim(0-epsilon,1+epsilon)
    plt.xticks(ticks=np.arange(len(names))+1, labels=names.tolist(), rotation=90)
    plt.tight_layout()
    plt.savefig('test_nn.png')
    print(names)
