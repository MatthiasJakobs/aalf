import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tsx.utils import get_device
from seedpy import fixedseed
from scipy.stats import zscore

# Define the LSTM-based model for time series forecasting
class DeepAR(nn.Module):
    def __init__(self, n_channel=1, hidden_size=50, num_layers=2, dropout=0.1, learning_rate=2e-3, max_epochs=100, batch_size=64, device=None):
        super(DeepAR, self).__init__()
        self.device = get_device() if device is None else device
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

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size

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
    def predict(self, x, covariates, length=100, num_samples=10):
        self.eval()
        x = torch.from_numpy(x).float().to(self.device)
        covariates = torch.from_numpy(covariates).float().to(self.device)
        batch_size = x.shape[0]
        input_size = x.shape[1]
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        preds = torch.zeros((batch_size, length, num_samples))

        # First, we get h0 and c0
        h0, c0 = self.initialize_hidden(batch_size)
        for t in range(input_size-1):
            _, _, h0, c0 = self(x[:, t].unsqueeze(1), h0, c0)

        # Next, do the actual prediction
        for j in range(num_samples):
            n_h0, n_c0 = h0.clone(), c0.clone()
            buffer = torch.cat([x[:, -1:].clone(), torch.zeros((batch_size, length, x.shape[-1])).to(self.device)], axis=1)
            for t in range(length):
                mu, sigma, n_h0, n_c0 = self(buffer[:, t].unsqueeze(1), n_h0, n_c0)
                dist = torch.distributions.normal.Normal(mu, sigma)
                sample = dist.sample().reshape(batch_size, 1)
                # TODO: Scaling? 
                preds[:, t, j] = sample
                buffer[:, t+1, 0] = sample.squeeze()
                buffer[:, t+1, 1:] = covariates[t] 

        return preds.numpy()

    # X.shape = (n_windows, L) or (n_windows, L, n_channels)
    # y is not used
    def fit(self, X, y=None):
        n_samples = X.shape[0]
        L  = X.shape[1]

        X = torch.from_numpy(X).float()
        y = X.clone()

        if len(X.shape) == 2:
            X = X.unsqueeze(-1)

        n_channels = X.shape[2]

        # Offset inputs by one 
        X = torch.cat([torch.zeros(n_samples, 1, n_channels), X[:, :-1]], axis=1)
        y = y[..., 0] # TODO: Better handling for multivariate data necessary

        X = X.to(self.device)
        y = y.to(self.device)

        # Batch data
        ds = torch.utils.data.TensorDataset(X, y)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        for epoch in range(self.max_epochs):
            self.train()
            epoch_loss = []
            for b_x, b_y in dl:
                optimizer.zero_grad()

                h0, c0 = self.initialize_hidden(b_x.shape[0])
                loss = 0

                for t in range(L):
                    mu, sigma, h0, c0 = self(b_x[:, t].unsqueeze(1).clone(), h0, c0)
                    ll = torch.distributions.normal.Normal(mu, sigma).log_prob(b_y[:, t]).mean()
                    loss += ll
                
                loss = -loss
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())

            print(f'Epoch [{epoch+1}/{self.max_epochs}], Loss: {np.mean(epoch_loss) / L:.4f}')

def generate_covariates(length, freq, start_date=None):
    def _zscore(X):
        if np.all(X[0] == X):
            return np.zeros((len(X)))
        return zscore(X)

    if start_date is None:
        start_date = '1970-01-01'

    time_series = pd.date_range(start=start_date, periods=length, freq=freq)

    weekday = _zscore(time_series.weekday.to_numpy())
    hour = _zscore(time_series.hour.to_numpy())
    month = _zscore(time_series.month.to_numpy())
    covariates = np.stack([weekday, hour, month]).T

    return covariates

def main():
    # Example input
    time_series = np.sin(np.linspace(0, 100, 1100))  # Just an example time series
    covariates = generate_covariates(len(time_series), freq='1min')
    time_series = np.concatenate([time_series[..., None], covariates], axis=1)

    X_train = time_series[:1000]
    y_test = time_series[1000:]

    # Prepare the data
    L = 50
    x_train = np.lib.stride_tricks.sliding_window_view(X_train, L, axis=0).copy().swapaxes(1, 2)

    with fixedseed([torch, np], 109228):
        model = DeepAR(n_channel=4, hidden_size=16, num_layers=1, dropout=0, max_epochs=20, learning_rate=2e-3)
        model.fit(x_train)

        print("Training complete.")
        test_input = x_train[-1:]
        length = 100
        # Since the covariates are assumed to be known at each time step we need to provide them for the values that need be predicted
        preds = model.predict(test_input, length=length, num_samples=10, covariates=y_test[:, 1:])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4))
    last = 500
    ax.plot(np.arange(last), X_train[-last:, 0], label='input')
    preds = preds.squeeze()
    ax.plot(np.arange(length)+last-1, preds.mean(-1), label='pred (mean)', color='b')
    ax.fill_between(np.arange(length)+last-1, preds.mean(-1) - preds.std(-1), preds.mean(-1) + preds.std(-1), color='b', alpha=0.1)
    fig.savefig('test.png')


if __name__ == '__main__':
    main()