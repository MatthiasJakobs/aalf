import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from selection import Oracle
from utils import rmse
from plotz import default_plot, COLORS
from config import DS_MAP

def plot_oracle_line(ax, ys, fint_preds, fcomp_preds, loss_fn=None, color=COLORS.blue, label=''):

    if loss_fn is None:
        loss_fn = rmse

    ps = np.linspace(0.001, 0.999, num=100)
    oracle_losses = []
    for p in ps:
        oracle = Oracle(p)
        n_datapoints = len(fint_preds)

        losses = []
        for ds_index in range(n_datapoints):
            oracle_indices = oracle.get_labels(ys[ds_index], fcomp_preds[ds_index], fint_preds[ds_index])
            oracle_preds = np.choose(oracle_indices, [fcomp_preds[ds_index], fint_preds[ds_index]])
            losses.append(loss_fn(ys[ds_index], oracle_preds))

        oracle_losses.append(np.mean(losses))

    ax.plot(ps, oracle_losses, color=color, label=label, linestyle='--') 
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel(r'$p = B/T$')
    ax.set_ylabel('RMSE')

    return ax

def main():
    #fig, axs = plt.subplots(2,2, layout='constrained')
    fig, axs = default_plot(subplots=(2,2), height_fraction=1.3)
    axs = axs.ravel()
    ds_names = ['australian_electricity_demand', 'nn5_daily_nomissing', 'pedestrian_counts', 'weather']
    for idx, ds_name in enumerate(ds_names):
        with open(f'preds/{ds_name}.pickle', 'rb') as f:
            preds = pickle.load(f)

        losses = pd.read_csv(f'results/{ds_name}.csv', index_col=0)
        losses = losses[['linear_rmse', 'fcnn_rmse', 'deepar_rmse']]
        losses = losses.rename({'linear_rmse': 'linear', 'fcnn_rmse': 'fcnn', 'deepar_rmse': 'deepar'}, axis=1).mean()

        # Plot both oracles
        axs[idx].scatter(0, losses['fcnn'], color=COLORS.green, marker='x', s=20, label='FCNN' if idx == 0 else '')
        axs[idx].scatter(0, losses['deepar'], color=COLORS.red, marker='x', s=20, label='DeepAR' if idx == 0 else '')
        axs[idx].scatter(1, losses['linear'], color=COLORS.blue, marker='x', s=20, label='Linear' if idx == 0 else '')

        axs[idx] = plot_oracle_line(axs[idx], ys=preds['y'], fint_preds=preds['linear'], fcomp_preds=preds['fcnn'], color=COLORS.green, label='Oracle FCNN-LIN' if idx == 0 else '')
        axs[idx] = plot_oracle_line(axs[idx], ys=preds['y'], fint_preds=preds['linear'], fcomp_preds=preds['deepar'], color=COLORS.red, label='Oracle Deepar-LIN' if idx == 0 else '')
        axs[idx].set_title(DS_MAP[ds_name])

    fig.legend(ncols=5, loc='center', bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout()
    fig.savefig('plots/loss_floor.pdf', bbox_inches='tight')

if __name__ == '__main__':
    main()