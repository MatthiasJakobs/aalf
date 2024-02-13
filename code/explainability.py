import numpy as np

from datasets import load_dataset
from utils import rmse
from evaluation import preprocess_data, load_models
from selection import run_v12

def extract_rules_rf(x, rf):
    x = x.reshape(1, -1)
    sample_id = 0

    feature_thresholds = np.zeros((x.shape[-1], 2))
    feature_thresholds[:, 0] = -np.inf
    feature_thresholds[:, 1] = np.inf

    for tree in rf.estimators_:

        feature = tree.tree_.feature
        threshold = tree.tree_.threshold

        node_indicator = tree.decision_path(x)
        leave_id = tree.apply(x)
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]

        for node_id in node_index:
            if leave_id[sample_id] == node_id:
                continue

            if (x[sample_id, feature[node_id]] <= threshold[node_id]):
                old = feature_thresholds[feature[node_id], 1]
                new = min(old, threshold[node_id])
                feature_thresholds[feature[node_id], 1] = new
            else:
                old = feature_thresholds[feature[node_id], 0]
                new = max(old, threshold[node_id])
                feature_thresholds[feature[node_id], 0] = new

    return feature_thresholds

def get_forest_boundaries(ds_name, ds_index, test_idx):
    L = 10
    X, horizons, _ = load_dataset(ds_name)
    X = X[ds_index]
    H = horizons[ds_index]

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocess_data(X, L, H)
    f_i, f_c = load_models(ds_name, ds_index)

    lin_preds_train = f_i.predict(x_train)
    lin_preds_val = f_i.predict(x_val)
    lin_preds_test = f_i.predict(x_test)

    nn_preds_train = f_c.predict(x_train)
    nn_preds_val = f_c.predict(x_val)
    nn_preds_test = f_c.predict(x_test)

    closeness_threshold = 1e-6

    lin_val_error = (lin_preds_val.squeeze()-y_val.squeeze())**2
    nn_val_error = (nn_preds_val.squeeze()-y_val.squeeze())**2
    lin_val_better = np.where(lin_val_error-nn_val_error <= closeness_threshold)[0]
    nn_val_better = np.array([idx for idx in range(len(lin_val_error)) if idx not in lin_val_better]).astype(np.int32)
    assert len(lin_val_better) + len(nn_val_better) == len(lin_val_error)

    lin_test_error = (lin_preds_test.squeeze()-y_test.squeeze())**2
    nn_test_error = (nn_preds_test.squeeze()-y_test.squeeze())**2
    lin_test_better = np.where(lin_test_error-nn_test_error <= closeness_threshold)[0]
    nn_test_better = np.array([idx for idx in range(len(lin_test_error)) if idx not in lin_test_better]).astype(np.int32)
    assert len(lin_test_better) + len(nn_test_better) == len(lin_test_error)

    lin_preds_val = lin_preds_val.squeeze()
    lin_preds_test = lin_preds_test.squeeze()
    nn_preds_val = nn_preds_val.squeeze()
    nn_preds_test = nn_preds_test.squeeze()
    y_val = y_val.squeeze()
    y_test = y_test.squeeze()

    # -------------

    # Last errors
    from selection import get_last_errors, oracle
    lin_errors, nn_errors = get_last_errors(lin_preds_train, nn_preds_train, y_train, lin_preds_val, nn_preds_val, y_val)
    train_last_errors = (lin_errors - nn_errors)

    lin_errors, nn_errors = get_last_errors(lin_preds_val, nn_preds_val, y_val, lin_preds_test, nn_preds_test, y_test)
    test_last_errors = (lin_errors - nn_errors)

    val_selection = oracle(lin_preds_val, nn_preds_val, y_val,  p=0.9)

    # Build X
    X_train = np.vstack([lin_preds_val-nn_preds_val, train_last_errors]).T
    X_train = np.concatenate([X_train, x_val], axis=1)

    X_test = np.vstack([lin_preds_test-nn_preds_test, test_last_errors]).T
    X_test = np.concatenate([X_test, x_test], axis=1)

    y_test = oracle(lin_preds_test, nn_preds_test, y_test, p=0.9)

    # Train model(s)
    from sklearn.ensemble import RandomForestClassifier
    from models import UpsampleEnsembleClassifier
    clf = UpsampleEnsembleClassifier(RandomForestClassifier, 1, random_state=12345, n_estimators=128)
    clf.fit(X_train, val_selection)

    print(y_test[test_idx], clf.predict(X_test)[test_idx])
    print(np.where(y_test == 0))

    clf = clf.estimators[0]
    thresholds = extract_rules_rf(X_test[test_idx], clf)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,1)
    ax.bar(np.arange(len(thresholds)), X_test[test_idx])
    for x_idx, (_min, _max) in enumerate(thresholds):
        ax.vlines(x_idx, _min, _max, linewidth=5, color='black')
    fig.savefig('test.png')

def get_local_explanation(ds_name, ds_index, test_idx):
    L = 10
    X, horizons, _ = load_dataset(ds_name)
    X = X[ds_index]
    H = horizons[ds_index]

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocess_data(X, L, H)
    f_i, f_c = load_models(ds_name, ds_index)

    lin_preds_train = f_i.predict(x_train)
    lin_preds_val = f_i.predict(x_val)
    lin_preds_test = f_i.predict(x_test)

    nn_preds_train = f_c.predict(x_train)
    nn_preds_val = f_c.predict(x_val)
    nn_preds_test = f_c.predict(x_test)

    closeness_threshold = 1e-6

    lin_val_error = (lin_preds_val.squeeze()-y_val.squeeze())**2
    nn_val_error = (nn_preds_val.squeeze()-y_val.squeeze())**2
    lin_val_better = np.where(lin_val_error-nn_val_error <= closeness_threshold)[0]
    nn_val_better = np.array([idx for idx in range(len(lin_val_error)) if idx not in lin_val_better]).astype(np.int32)
    assert len(lin_val_better) + len(nn_val_better) == len(lin_val_error)

    lin_test_error = (lin_preds_test.squeeze()-y_test.squeeze())**2
    nn_test_error = (nn_preds_test.squeeze()-y_test.squeeze())**2
    lin_test_better = np.where(lin_test_error-nn_test_error <= closeness_threshold)[0]
    nn_test_better = np.array([idx for idx in range(len(lin_test_error)) if idx not in lin_test_better]).astype(np.int32)
    assert len(lin_test_better) + len(nn_test_better) == len(lin_test_error)

    lin_preds_val = lin_preds_val.squeeze()
    lin_preds_test = lin_preds_test.squeeze()
    nn_preds_val = nn_preds_val.squeeze()
    nn_preds_test = nn_preds_test.squeeze()
    y_val = y_val.squeeze()
    y_test = y_test.squeeze()

    print(test_idx in lin_test_better) 
    print(lin_test_better[:20])
    phi = f_i.coef_.squeeze()
    phi_0 = f_i.intercept_[0]

    n_features = len(phi)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,1)
    phi_x = x_test[test_idx].squeeze()[:n_features] * phi
    print(phi_0)
    print(phi_x)
    left = [phi_0] + [phi_0 + phi_x[:(i+1)].sum() for i in range(n_features-1)]
    color = ['red' if phi_x[i] >= 0 else 'blue' for i in range(n_features)]
    axs.set_xlim(-0.31, 0.12)
    rects = axs.barh(np.arange(n_features), phi_x, left=left, color=color, tick_label='bla')
    axs.set_yticks(np.arange(n_features), labels=[r'$x_{t-' + str(n_features-idx-1) + r'}$' for idx in range(n_features-1)] + [f'$x_t$'])

    for i, rect in enumerate(rects):
        width = rect.get_width()
        width = left[i] + width
        print(i, width)
        if (width-left[i]) < 0:
            xloc = -20
        else:
            xloc = 20
            
        clr = 'black'
        align = 'center'

        yloc = rect.get_y() + rect.get_height() / 2
        axs.annotate(f'{phi_x[i]:.4f}', xy=(width, yloc), xytext=(xloc, 0),
                            textcoords="offset points",
                            ha=align, va='center',
                            color=clr, weight='regular', clip_on=True)
        

    fig.tight_layout()
    fig.savefig('test.png')



if __name__ == '__main__':
    #get_forest_boundaries('kdd_cup_nomissing', 0, 1)
    #get_forest_boundaries('kdd_cup_nomissing', 0, 2030)
    #get_local_explanation('kdd_cup_nomissing', 0, 3)
    get_local_explanation('kdd_cup_nomissing', 0, 23)