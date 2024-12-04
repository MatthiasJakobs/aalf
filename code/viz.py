import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from selection import Oracle
from utils import rmse
from plotz import default_plot, COLORS
from config import DS_MAP, ALL_DATASETS

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

def plot_loss_floor():
    fig, axs = default_plot(subplots=(3,2), height_fraction=1.3)
    axs = axs.ravel()
    ds_names = ALL_DATASETS
    for idx, ds_name in enumerate(ds_names):
        with open(f'preds/{ds_name}.pickle', 'rb') as f:
            preds = pickle.load(f)

        losses = pd.read_csv(f'results/basemodel_losses/{ds_name}.csv', index_col=0)
        losses = losses[['linear_rmse', 'fcnn_rmse', 'deepar_rmse', 'cnn_rmse']]
        losses = losses.rename({'linear_rmse': 'linear', 'fcnn_rmse': 'fcnn', 'deepar_rmse': 'deepar', 'cnn_rmse': 'cnn'}, axis=1).mean()

        # Plot both oracles
        axs[idx].scatter(0, losses['fcnn'], color=COLORS.green, marker='x', s=20, label='FCNN' if idx == 0 else '')
        axs[idx].scatter(0, losses['deepar'], color=COLORS.red, marker='x', s=20, label='DeepAR' if idx == 0 else '')
        axs[idx].scatter(0, losses['cnn'], color=COLORS.orange, marker='x', s=20, label='CNN' if idx == 0 else '')
        axs[idx].scatter(1, losses['linear'], color=COLORS.blue, marker='x', s=20, label='AR' if idx == 0 else '')

        axs[idx] = plot_oracle_line(axs[idx], ys=preds['test']['y'], fint_preds=preds['test']['linear'], fcomp_preds=preds['test']['fcnn'], color=COLORS.green, label=r'$\mathcal{O}(\text{AR},\text{FCNN})$' if idx == 0 else '')
        axs[idx] = plot_oracle_line(axs[idx], ys=preds['test']['y'], fint_preds=preds['test']['linear'], fcomp_preds=preds['test']['deepar'], color=COLORS.red, label=r'$\mathcal{O}(\text{AR},\text{DeepAR})$' if idx == 0 else '')
        axs[idx] = plot_oracle_line(axs[idx], ys=preds['test']['y'], fint_preds=preds['test']['linear'], fcomp_preds=preds['test']['cnn'], color=COLORS.orange, label=r'$\mathcal{O}(\text{AR},\text{CNN})$' if idx == 0 else '')
        axs[idx].set_title(DS_MAP[ds_name])

    fig.legend(ncols=7, loc='center', columnspacing=1.0, handletextpad=0.4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout()
    fig.savefig('plots/loss_floor.pdf', bbox_inches='tight')

def plot_optimum_example():
    fig, axs = default_plot(subplots=(1,2), height_fraction=1, sharey=True)
    axs = axs.ravel()
    
    # Left side: ||s|| = B
    # Right side: ||s|| >= B

    B = 5

    y = np.array([-0.17, -0.1, -0.05, 0.01, 0.04, 0.09, 0.1, 0.14, 0.15, 0.17])
    xlabel = [fr'${t}$' for t in range(1, len(y)+1)]
    negatives = np.where(y < 0)[0]
    positives = np.where(y >= 0)[0]
    axs[0].grid(zorder=0)
    axs[0].bar(x=np.arange(len(y))[negatives], height=y[negatives], color=COLORS.blue, zorder=3)
    axs[0].bar(x=np.arange(len(y))[positives], height=y[positives], color=COLORS.red, zorder=3)
    axs[0].axvline(x=B+0.51, color='grey', linestyle='--', alpha=0.5, lw=0.75)
    axs[0].set_xticks(ticks=np.arange(len(y)), labels=xlabel)
    axs[0].set_xlabel(fr'$\pi(t)$')
    axs[0].set_ylabel(fr'$\ell(\pi(t))$')

    y = np.array([-0.17, -0.15, -0.1, -0.09, -0.06, -0.04, -0.02, 0.04, 0.07, 0.1])
    xlabel = [fr'${t}$' for t in range(1, len(y)+1)]
    negatives = np.where(y < 0)[0]
    positives = np.where(y >= 0)[0]
    axs[1].grid(zorder=0)
    axs[1].bar(x=np.arange(len(y))[negatives], height=y[negatives], color=COLORS.blue, zorder=3)
    axs[1].bar(x=np.arange(len(y))[positives], height=y[positives], color=COLORS.red, zorder=3)
    axs[1].axvline(x=B+0.51, color='grey', linestyle='--', alpha=0.5, lw=0.75, label=fr'$B={B+1}$')
    axs[1].set_xticks(ticks=np.arange(len(y)), labels=xlabel)
    axs[1].set_xlabel(fr'$\pi(t)$')
    axs[1].set_ylabel(fr'$\ell(\pi(t))$')
    axs[1].legend()

    fig.tight_layout()
    fig.savefig('plots/optimum_example.pdf', bbox_inches='tight')


if __name__ == '__main__':
    plot_loss_floor()
    plot_optimum_example()