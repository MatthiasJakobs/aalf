import pandas as pd

def main():
    ds_names = ['weather', 'pedestrian_counts', 'nn5', 'web_traffic', 'kdd_cup_nomissing']
    #ds_names = ['pedestrian_counts']
    summary = {}
    basemodels = ['linear', 'LastValue', 'MeanValue', 'trf-128', 'fcn', 'svr', 'local_cnn', 'local_cnn_2', 'HetEnsemble-median']

    for ds_name in ds_names:
        df = pd.read_csv(f'results/{ds_name}_test.csv', header=0)
        df = df.set_index('dataset_names')
        df = df[basemodels]
        mean_rmse = df.mean(axis=0)

        summary[ds_name] = mean_rmse

    print(pd.DataFrame(summary))

if __name__ == '__main__':
    main()