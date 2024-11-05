from gluonts.dataset.common import ListDataset
from gluonts.dataset.repository import get_dataset
from config import all_datasets, tsx_to_gluon
from tsx.datasets import split_proportion

def get_train_data(dataset):
    d  = dataset.train

    Xs_train, Xs_val, Xs_test = [], [], []
    for entry in d:
        X = entry['target']
        X_train, X_val, X_test = split_proportion(X, (0.5, 0.25, 0.25))
        Xs_train.append(X_train)
        Xs_val.append(X_val)
        Xs_test.append(X_test)
    return Xs_train, Xs_val, Xs_test

if __name__ == '__main__':
    total = 0
    for ds_name in all_datasets:
        ds = get_dataset(tsx_to_gluon[ds_name])
        Xs, _, _ = get_train_data(ds)
        print(ds_name, len(Xs))
        total += len(Xs)
    print('---')
    print('total', total)