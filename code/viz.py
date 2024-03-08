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

def show_empirical_selection_performance_table():
    ds_names = ['pedestrian_counts', 'kdd_cup_nomissing', 'weather', 'web_traffic']
    cols = sum([[f'v12_{p}', f'NewOracle{int(p*100)}'] for p in [0.5, 0.7, 0.9, 0.95, 0.99]], [])
    all_series_mean = []
    all_series_std = []
    for ds_name in ds_names:
        df = pd.read_csv(f'results/{ds_name}_selection.csv').set_index('dataset_names')
        all_series_mean.append(df[cols].mean(axis=0))
        all_series_std.append(df[cols].std(axis=0))

    df_mean = pd.concat(all_series_mean, axis=1).T
    df_mean.index = ds_names
    df_mean = df_mean.applymap(lambda x: f'{x:.2f}')

    df_std = pd.concat(all_series_std, axis=1).T
    df_std.index = ds_names
    df_std = df_std.applymap(lambda x: f'{x:.2f}')

    print(df_mean)
    print(df_std)

    combined_df = df_mean.astype('str') + ' \pm ' + df_std.astype('str')
    combined_df = combined_df.applymap(lambda e: f'${e}$')
    combined_df = combined_df.rename(columns=TREATMENT_DICT, index=DATASET_DICT_SMALL)
    tex = combined_df.to_latex()
    tex_list = tex.splitlines()
    tex_list = ['\\begin{tabular*}{\\linewidth}{@{\extracolsep{\\fill}} '+ 'l'*(len(cols)+1) + '}'] + tex_list[1:-1] + ['\\end{tabular*}']
    tex = '\n'.join(tex_list)

    with open('results/sel_table.tex', 'w+') as _f:
        _f.write(tex)

def show_empirical_selection_performance_boxplot():
    ds_names = ['pedestrian_counts', 'kdd_cup_nomissing', 'weather', 'web_traffic']
    ps = [0.5, 0.6, 0.7, 0.8, 0.9]
    cols = sum([[f'v12_{p}', f'NewOracle{int(p*100)}'] for p in ps], [])

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=get_figsize('LNCS', subplots=(2,2)))

    offset = 0.15
    width = 0.2
    positions = sum([[idx - offset, idx + offset] for idx in range(len(cols)//2)], [])

    flierprops = dict(marker='+', markersize=1, alpha=0.5)
    boxprops = dict(linewidth=0.5)
    capprops = dict(linewidth=0.5)
    medianprops = dict()
    whiskerprops = dict(linewidth=0.5)
    showflier = False

    for idx, ds_name in enumerate(ds_names):
        df = pd.read_csv(f'results/{ds_name}_selection.csv').set_index('dataset_names')[cols]
        ax = axs.ravel()[idx]
        for c_idx, col_name in enumerate(cols):
            if c_idx % 2 == 0:
                boxprops.update({'facecolor': 'C6'})
            else:
                boxprops.update({'facecolor': 'C7'})
            ax.boxplot(df[col_name], patch_artist=True, positions=[positions[c_idx]], widths=width, manage_ticks=False, flierprops=flierprops, boxprops=boxprops, capprops=capprops, whiskerprops=whiskerprops, showfliers=showflier, medianprops=medianprops)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(np.arange(5), ps)
        ax.set_title(DATASET_DICT[ds_name])
        ax.grid(alpha=0.5, zorder=0, markevery=0.25)

    fig.supxlabel(r'$p$')
    fig.supylabel(r'Mean Selection of $f_i$ in percent')

    fig.tight_layout()
    fig.subplots_adjust(top=0.83)
    fig.legend(bbox_to_anchor=(0.5, 1), loc='upper center', ncol=2, handles=[mpatches.Patch(color='C6', label='AALF'), mpatches.Patch(color='C7', label='Oracle')])
    fig.savefig('plots/selection_boxplot.pdf')

if __name__ == '__main__':
    #plot_all_selection_percentage()
    #plot_selection_percentage('weather', drop_columns=['selBinom0.9', 'selBinom0.95', 'selBinom0.99', 'v4_0.5', 'v5', 'v8', 'test_1.2', 'v10'])
    #plot_selection_performance(['v9', 'v10', 'v11', 'test_1.0'])
    #plot_selection_percentage_single('web_traffic', ['v11_0.7', 'v10_0.7'])
    plot_selection_performance(['v12', 'ade', 'dets', 'knnroc', 'oms'])
    #plot_global_feature_importance()
    show_empirical_selection_performance_table()
    show_empirical_selection_performance_boxplot()

