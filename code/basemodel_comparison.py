import pandas as pd

def main():
    ds_names = ['weather', 'pedestrian_counts', 'web_traffic', 'kdd_cup_nomissing', 'nn5']
    summary = {}
    basemodels = ['linear', 'LastValue', 'MeanValue', 'trf-raw', 'global_cnn']

    for ds_name in ds_names:
        df = pd.read_csv(f'results/{ds_name}_test.csv', header=0)
        df = df.set_index('dataset_names')
        df = df[basemodels]
        mean_rmse = df.mean(axis=0)

        summary[ds_name] = mean_rmse

    print(pd.DataFrame(summary))

if __name__ == '__main__':
    main()