import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the LSTM-based model for time series forecasting
class DeepAR(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.1, device='cpu'):
        super(DeepAR, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.mean_estimator = nn.Linear(hidden_size*num_layers, 1)
        self.var_estimator = nn.Linear(hidden_size*num_layers, 1)

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
    def predict(self, x, length=100, num_samples=10):
        self.eval()
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
            buffer = torch.cat([x[:, -1:], torch.zeros((batch_size, length, x.shape[-1]))], axis=1)
            for t in range(length):
                mu, sigma, n_h0, n_c0 = self(buffer[:, t].unsqueeze(1), n_h0, n_c0)
                dist = torch.distributions.normal.Normal(mu, sigma)
                sample = dist.sample().reshape(batch_size, 1)
                # TODO: Scaling? 
                preds[:, t, j] = sample
                buffer[:, t+1] = sample

        return preds.numpy()

def main():
    # Example input
    time_series = np.sin(np.linspace(0, 100, 1000))  # Just an example time series

    # Prepare the data
    input_size = 50
    labels = np.lib.stride_tricks.sliding_window_view(time_series, input_size)
    num_samples = labels.shape[0]
    labels = torch.tensor(labels, dtype=torch.float32)  # Shape: (num_samples, input_size)
    inputs = torch.cat([torch.zeros((num_samples, 1)), labels[:, :-1]], axis=1).unsqueeze(-1) # Shape: (num_samples, input_size, 1)

    # Initialize the model, loss function, and optimizer
    model = DeepAR(input_size=1, hidden_size=16, num_layers=1, dropout=0)
    #criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    print(inputs.shape)
    print(labels.shape)
    # Training loop

    epochs = 400

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        h0, c0 = model.initialize_hidden(inputs.shape[0])
        loss = 0

        for t in range(input_size):
            mu, sigma, h0, c0 = model(inputs[:, t].unsqueeze(1).clone(), h0, c0)
            ll = torch.distributions.normal.Normal(mu, sigma).log_prob(labels[:, t]).mean()
            loss += ll
        
        loss = -loss
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item() / input_size:.4f}')

    print("Training complete.")
    test_input = inputs[-1:]
    print(test_input.shape)
    length = 100
    preds = model.predict(test_input, length=length, num_samples=100)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4))
    last = 500
    ax.plot(np.arange(last), time_series[-last:], label='input')
    preds = preds.squeeze()
    ax.plot(np.arange(length)+last-1, preds.mean(-1), label='pred (mean)', color='b')
    ax.fill_between(np.arange(length)+last-1, preds.mean(-1) - preds.std(-1), preds.mean(-1) + preds.std(-1), color='b', alpha=0.1)
    fig.savefig('test.png')


if __name__ == '__main__':
    main()