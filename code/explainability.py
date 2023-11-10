import numpy as np
import torch
import pickle
import tqdm

from models import Ensemble, PyTorchEnsemble, PyTorchLinear
from captum.attr import DeepLiftShap

def get_explanations(model, X, y, background, batch_size=16):
    explainer = DeepLiftShap(model)
    explanations = []

    for i in tqdm.trange(0, len(X), batch_size):
        explanations.append(explainer.attribute(X[i:(i+batch_size)], background, additional_forward_args=(y[i:i+batch_size],)).detach().cpu().numpy())

    return np.concatenate(explanations, axis=0)

def main():
    with open('models/weather/0/nn.pickle', 'rb') as f:
        old_ensemble = pickle.load(f)

    ensemble = PyTorchEnsemble(old_ensemble)
    X = torch.rand((16, 10))
    y = torch.rand((16))
    background = torch.rand((64, 10))

    explanations = get_explanations(ensemble, X, y, background)
    print('nn', explanations.shape)

    # ----------------------------------
    with open('models/weather/0/linear.pickle', 'rb') as f:
        lin = pickle.load(f)

    model = PyTorchLinear(lin)
    explanations = get_explanations(model, X, y, background)
    print('lin', explanations.shape)

if __name__ == '__main__':
    main()