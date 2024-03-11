import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sktime.clustering.k_medoids import TimeSeriesKMedoids
from tslearn.utils import to_time_series_dataset
from tsx.model_selection import ROC_Member
from sklearn.metrics import silhouette_score
from cdd_plots import TREATMENT_DICT, DATASET_DICT, DATASET_DICT_SMALL

TEMPLATE_WIDTHS = {
    'LNCS': 347.12354
}

def get_figsize(template, height_scale=1, subplots=(1,1)):

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "text.latex.preamble": r'\usepackage{bm}',
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 8,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8
    }

    plt.rcParams.update(tex_fonts)
    width = TEMPLATE_WIDTHS[template]
    fig_width_pt = width

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1]) * height_scale

    fig_dim = (fig_width_in, fig_height_in)
    return fig_dim

def _plot_single_selection_performance(ax, idx, ds_name, methods, s=12):
    errors = pd.read_csv(f'results/{ds_name}_test.csv')
    selections = pd.read_csv(f'results/{ds_name}_selection.csv')

    # Draw horizontal line where NN error is 
    ax.axhline(errors['nn'].mean(), linestyle='--', linewidth=1, color='black', alpha=0.3, zorder=0)

    ax.scatter(1, errors['nn'].mean(), s=s, label=r'$f_c$' if idx == 0 else '', zorder=5)
    ax.scatter(0, errors['linear'].mean(), s=s, label=r'$f_i$' if idx == 0 else '', zorder=5)

    for method in methods:
        label = TREATMENT_DICT.get(method, method)
        if method == 'v12':
            color = 'C6'
            ps = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
            sel = [1-selections[f'v12_{p}'].mean() for p in ps]
            err = [errors[f'v12_{p}'].mean() for p in ps]

            ax.plot(sel, err, c=color, alpha=.7, zorder=5)
            ax.scatter(sel, err, c=color, s=s, label=label if idx == 0 else '', zorder=5)
            continue

        # Try all other methods
        try:
            sel = 1-selections[method].mean()
            err = errors[method].mean()
            ax.plot(sel, err, alpha=.7, zorder=5)
            ax.scatter(sel, err, s=s+5, marker='*', label=label if idx == 0 else '', zorder=5)
        except KeyError:
            print('Method', method, 'not in results, skipping')
            continue


    ps = [50, 60, 70, 80, 90, 95, 99]
    sel = [1-selections[f'NewOracle{p}'].mean() for p in ps]
    err = [errors[f'NewOracle{p}'].mean() for p in ps]
    color = 'C7'
    ax.plot(sel, err, c=color, alpha=.7, zorder=5)
    ax.scatter(sel, err, marker='+', s=s+5, c=color, label='Oracle' if idx == 0 else '', zorder=5)

    ax.grid(alpha=0.5, zorder=0)
    ax.set_title(DATASET_DICT.get(ds_name, ds_name))
    return ax

def plot_selection_percentage_single(ds_name, methods):
    fig, ax = plt.subplots(1,1, sharex=True)
    ax = _plot_single_selection_performance(ax, 0, ds_name, methods)
    fig.supylabel(r'Mean RMSE ($\downarrow$)')
    fig.supxlabel(r'Mean Selection of $f_c$ in percent ($\downarrow$)')
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)
    fig.legend(bbox_to_anchor=(0.5, 1), loc='upper center', ncol=(len(methods)+3)//2)
    fig.savefig(f'plots/scatter_{ds_name}.png')

def plot_selection_performance(methods):
    ds_names = ['weather', 'pedestrian_counts', 'web_traffic', 'kdd_cup_nomissing']
    fig, axs = plt.subplots(2,2, sharex=True, figsize=get_figsize('LNCS', subplots=(2,2)))
    for idx, ds_name in enumerate(ds_names):
        ax = axs.ravel()[idx]
        ax = _plot_single_selection_performance(ax, idx, ds_name, methods)
    fig.supylabel(r'Mean RMSE ($\downarrow$)')
    fig.supxlabel(r'Mean Selection of $f_c$ in percent ($\downarrow$)')
    fig.tight_layout()
    fig.subplots_adjust(top=0.78)
    fig.legend(bbox_to_anchor=(0.5, 1), loc='upper center', ncol=(len(methods)+3)//2)
    fig.savefig('plots/scatter.png')
    fig.savefig('plots/scatter.pdf')
        
def plot_global_feature_importance():
    fig, axs = plt.subplots(2,2, sharex=True, sharey=True, figsize=get_figsize('LNCS', subplots=(2,2)))
    ds_names = ['weather', 'pedestrian_counts', 'web_traffic', 'kdd_cup_nomissing']
    for idx, ds_name in enumerate(ds_names):
        ax = axs.ravel()[idx]
        df = pd.read_csv(f'results/{ds_name}_gfi.csv').set_index('dataset_names')
        X = df.dropna(axis=0).to_numpy()

        mus = X.mean(axis=0)
        stds = X.std(axis=0)

        ax.bar(0, mus[0], yerr=stds[0], color='C3', label=r'$(\hat{y}_{t,c}-\hat{y}_{t,i})$' if idx == 0 else '', capsize=2) 
        ax.bar(1, mus[1], yerr=stds[1], color='C2', label=r'$(\hat{e}_{t,c}-\hat{e}_{t,i})$' if idx == 0 else '', capsize=2) 
        ax.bar(np.arange(10)+2, mus[2:12], yerr=stds[2:12], label=r'$\bm{x}_t$' if idx == 0 else '', capsize=2) 

        ax.set_title(DATASET_DICT.get(ds_name, ds_name))
        ax.set_xticks([])

    fig.supylabel('Mean Global Feature Importance')
    fig.tight_layout()
    fig.subplots_adjust(top=0.83)
    fig.legend(bbox_to_anchor=(0.5, 1), loc='upper center', ncol=3)
    fig.savefig(f'plots/gfi.pdf')

def show_empirical_selection_performance_graph():
    ds_names = ['pedestrian_counts', 'kdd_cup_nomissing', 'weather', 'web_traffic']
    heightscale = 0.5
    fig, axs = plt.subplots(4, 1, sharex=False, sharey=True, figsize=get_figsize('LNCS', subplots=(4,1), height_scale=heightscale))
    for i, ds_name in enumerate(ds_names):
        axs[i] = _show_empirical_selection_performance_graph(ds_name, heightscale=heightscale, ax=axs[i])

    # Custom legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = list(set(labels))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]

    # Create legend
    fig.supxlabel(r'$p$')
    fig.supylabel(r'Mean Selection of $f_i$ in percent')
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.legend(unique_handles, unique_labels, bbox_to_anchor=(0.5, 1), loc='upper center', ncol=2)
    fig.savefig('plots/selection_conf_all.pdf')

def _show_empirical_selection_performance_graph(ds_name, heightscale=0.8, ax=None):
    ps = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    cols = sum([[f'v12_{p}', f'NewOracle{int(p*100)}'] for p in ps], [])
    df = pd.read_csv(f'results/{ds_name}_selection.csv').set_index('dataset_names')
    df = df[cols]

    set_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=get_figsize('LNCS', subplots=(1,1), height_scale=heightscale))
        set_fig = True

    # v12
    color = 'C6'
    col_names = [col_name for col_name in df.columns if col_name.startswith('v12') ]
    mus = df[col_names].mean(axis=0).to_numpy().astype(np.float32)
    stds = df[col_names].std(axis=0).to_numpy().astype(np.float32)
    label = 'AALF'
    ax.errorbar(ps, mus, yerr=stds, capsize=3, color=color, label=label)

    # Oracle
    color = 'C7'
    col_names = [col_name for col_name in df.columns if col_name.startswith('New') ]
    mus = df[col_names].mean(axis=0).to_numpy().astype(np.float32)
    stds = df[col_names].std(axis=0).to_numpy().astype(np.float32)
    label = 'Oracle'
    ax.errorbar(ps, mus, yerr=stds, capsize=3, color=color, label=label)

    ax.set_ylim(-0.05, 1.15)
    ax.set_title(DATASET_DICT[ds_name])
    ax.grid(alpha=0.5, zorder=0)
    ax.set_xticks(ps, ps)
    if set_fig:
        ax.set_xlabel(r'$p$')
        ax.set_ylabel(r'Mean Selection of $f_i$ in percent')

    if set_fig:
        fig.tight_layout()
        fig.legend(loc='center right')
        fig.savefig(f'plots/selection_conf_{ds_name}.pdf')
    return ax

if __name__ == '__main__':
    plot_selection_performance(['v12', 'ade', 'dets', 'knnroc', 'oms'])
    _show_empirical_selection_performance_graph('pedestrian_counts')
    _show_empirical_selection_performance_graph('weather')
    _show_empirical_selection_performance_graph('web_traffic')
    _show_empirical_selection_performance_graph('kdd_cup_nomissing')
    show_empirical_selection_performance_graph()
    plot_global_feature_importance()

