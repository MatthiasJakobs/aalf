import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans
from sktime.clustering.k_medoids import TimeSeriesKMedoids
from tslearn.utils import to_time_series_dataset
from tsx.model_selection import ROC_Member
from sklearn.metrics import silhouette_score

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

def plot_selection_percentage(ds_name):
    df = pd.read_csv(f'results/{ds_name}_selection.csv')
    names = df.columns[1:]
    plt.figure()
    plt.violinplot(df.iloc[:, 1:].to_numpy(), showmedians=True)
    #plt.boxplot(df.iloc[:, 1:].to_numpy())
    epsilon = 0.05
    plt.ylim(0-epsilon,1+epsilon)
    plt.xticks(ticks=np.arange(len(names))+1, labels=names.tolist(), rotation=90)
    plt.tight_layout()
    plt.savefig('test.png')
    print(names)
    
if __name__ == '__main__':
    plot_selection_percentage('weather')