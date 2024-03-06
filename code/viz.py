import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans
from sktime.clustering.k_medoids import TimeSeriesKMedoids
from tslearn.utils import to_time_series_dataset
from tsx.model_selection import ROC_Member
from sklearn.metrics import silhouette_score
from selection import selection_oracle_percent
from cdd_plots import TREATMENT_DICT, DATASET_DICT

TEMPLATE_WIDTHS = {
    'LNCS': 347.12354
}

def get_figsize(template, height_scale=1, subplots=(1,1)):

    # Using seaborn's style
    #plt.style.use('seaborn')

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
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

def cluster_rocs(rocs):
    # Find cluster with minimal inertia
    tslearn_formatted = to_time_series_dataset([r.x for r in rocs])
    l, u = 2, 5
    best_silhouette = -1
    best_k = l
    for k in range(l, u):
        #km = TimeSeriesKMeans(n_clusters=k, n_init=10, n_jobs=-1, random_state=20231113+k)
        km = TimeSeriesKMedoids(n_clusters=k, metric='euclidean', random_state=20231113+k)
        km.fit(np.stack([r.x for r in rocs]))
        silhouette_ = silhouette_score([r.x for r in rocs], km.labels_, metric='euclidean')
        if silhouette_ > best_silhouette:
            best_silhouette = silhouette_
            best_k = k

    #km = TimeSeriesKMeans(n_clusters=best_k, random_state=20231113+k)
    km = TimeSeriesKMedoids(n_clusters=best_k, metric='euclidean', random_state=20231113+k)
    km.fit(np.stack([r.x for r in rocs]))
    centers = km.cluster_centers_
    return centers, best_k

def plot_rocs(roc_lin, roc_complex, save_path):
    # Normalize data
    norm_roc_lin = []
    for r in roc_lin:
        std = np.std(r.x)
        mu = np.mean(r.x)
        if std <= 0.01:
            std = 1
        _x = ((r.x - mu) / std)
        norm_roc_lin.append(ROC_Member(_x, (r.y - mu) / std, r.indices))

    norm_roc_ens = []
    for r in roc_complex:
        std = np.std(r.x)
        mu = np.mean(r.x)
        if std <= 0.01:
            std = 1
        _x = ((r.x - mu) / std)
        norm_roc_ens.append(ROC_Member(_x, (r.y - mu) / std, r.indices))

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    centers, best_k = cluster_rocs(norm_roc_lin)
    centers = centers.squeeze()
    print('lin', len(centers), best_k)
    # Plot linear
    for cc in centers:
        axs[0].plot(cc)
        axs[0].scatter(np.arange(len(cc)), cc)
        axs[0].set_title(f'Linear RoCs, k={best_k}')

    centers, best_k = cluster_rocs(norm_roc_ens)
    centers = centers.squeeze()
    print('ens', len(centers), best_k)

    # Plot ensemble
    for cc in centers:
        axs[1].plot(cc)
        axs[1].scatter(np.arange(len(cc)), cc)
        axs[1].set_title(f'Ensemble RoCs, k={best_k}')


    fig.tight_layout()
    #fig.savefig(save_path)
    fig.savefig('test.png')
    exit()

def plot_selection_percentage(ds_name, drop_columns=None):
    df = pd.read_csv(f'results/{ds_name}_selection.csv')
    if drop_columns is not None:
        df = df.drop(columns=drop_columns)
    names = df.columns[1:]
    plt.figure()
    plt.axhline(0.5, ls='--', alpha=0.5, color='black')
    plt.violinplot(df.iloc[:, 1:].to_numpy(), showmedians=True)
    #plt.boxplot(df.iloc[:, 1:].to_numpy())
    epsilon = 0.05
    plt.ylim(0-epsilon,1+epsilon)
    plt.xticks(ticks=np.arange(len(names))+1, labels=names.tolist(), rotation=90)
    plt.ylabel(r'$p_{linear}$')
    plt.title(ds_name)
    plt.tight_layout()
    plt.savefig(f'plots/selection_{ds_name}.png')

def plot_all_selection_percentage():
    ds_names = ['web_traffic', 'london_smart_meters_nomissing', 'kdd_cup_nomissing', 'pedestrian_counts', 'weather']
    drop_columns = ['v4_0.5_calibrated', 'v4_0.5', 'v5', 'v8', 'selBinom0.9', 'selBinom0.95', 'selBinom0.99']
    width = 6
    height = len(ds_names)/2 * width
    fig, axs = plt.subplots(nrows=len(ds_names), ncols=1, sharex=True, sharey=True, figsize=(width, height))
    names = None
    for row_idx in range(len(ds_names)):
        ds_name = ds_names[row_idx]
        df = pd.read_csv(f'results/{ds_name}_selection.csv')
        df = df.drop(columns=drop_columns, errors='ignore')
        if names is None:
            names = df.columns[1:]
            print(names)

        ax = axs[row_idx]
        ax.set_title(ds_name)
        ax.set_ylabel(r'$p_{linear}$')
        ax.axhline(0.5, ls='--', alpha=0.5, color='black')
        ax.violinplot(df.iloc[:, 1:].to_numpy(), showmedians=True)

        epsilon = 0.05
        ax.set_ylim(0-epsilon,1+epsilon)
        ax.set_xticks(ticks=np.arange(len(names))+1, labels=names.tolist(), rotation=60)

    fig.tight_layout()
    fig.savefig('test_all.png')

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
            ps = [0.5, 0.6, 0.7, 0.8, 0.9]
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


    ps = [50, 60, 70, 80, 90]
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

        ax.bar(0, mus[0], yerr=stds[0], color='C3', label=r'$(\hat{y}_{t,c}-\hat{y}_{t,i})$' if idx == 0 else '') 
        ax.bar(1, mus[1], yerr=stds[1], color='C2', label=r'$(\hat{e}_{t,c}-\hat{e}_{t,i})$' if idx == 0 else '') 
        ax.bar(np.arange(X.shape[-1]-2)+2, mus[2:], yerr=stds[2:], label=r'$x_t$' if idx == 0 else '') 

        ax.set_title(DATASET_DICT.get(ds_name, ds_name))
        ax.set_xticks([])

    fig.supylabel('Mean Global Feature Importance')
    fig.tight_layout()
    fig.subplots_adjust(top=0.83)
    fig.legend(bbox_to_anchor=(0.5, 1), loc='upper center', ncol=3)
    fig.savefig(f'plots/gfi.pdf')

def show_empirical_selection_performance():
    ds_names = ['pedestrian_counts', 'kdd_cup_nomissing', 'weather', 'web_traffic']
    cols = sum([[f'v12_{p}', f'NewOracle{int(p*100)}'] for p in [0.5, 0.7, 0.9]], [])
    all_series = []
    for ds_name in ds_names:
        df = pd.read_csv(f'results/{ds_name}_selection.csv').set_index('dataset_names')
        all_series.append(df[cols].mean(axis=0))

    df = pd.concat(all_series, axis=1).T
    df.index = ds_names
    print(df)
    #print(df[[f'v12_{p}', f'NewOracle{int(p*100)}']].mean(axis=0))


if __name__ == '__main__':
    #plot_all_selection_percentage()
    #plot_selection_percentage('weather', drop_columns=['selBinom0.9', 'selBinom0.95', 'selBinom0.99', 'v4_0.5', 'v5', 'v8', 'test_1.2', 'v10'])
    #plot_selection_performance(['v9', 'v10', 'v11', 'test_1.0'])
    #plot_selection_percentage_single('web_traffic', ['v11_0.7', 'v10_0.7'])
    #plot_selection_performance(['v12', 'ade', 'dets', 'knnroc', 'oms'])
    #plot_global_feature_importance()
    show_empirical_selection_performance()

