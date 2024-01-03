import pandas as pd

def get_dataset_statistics():
    ds_names = ['web_traffic', 'london_smart_meters_nomissing', 'kdd_cup_nomissing', 'weather', 'pedestrian_counts']
    total = 0
    for ds_name in ds_names:
        df = pd.read_csv(f'results/{ds_name}_test.csv')
        n_datapoints = len(df)
        total += n_datapoints
        print(ds_name, n_datapoints)
    print('Total:', total)

if __name__ == '__main__':
    get_dataset_statistics()