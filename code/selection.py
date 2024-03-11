import numpy as np
from sklearn.ensemble import RandomForestClassifier
from models import UpsampleEnsembleClassifier

def get_last_errors(lin_preds_val, nn_preds_val, y_val, lin_preds_test, nn_preds_test, y_test):
    last_preds_lin = np.concatenate([lin_preds_val[-1].reshape(-1), lin_preds_test[:-1].reshape(-1)])
    last_preds_nn = np.concatenate([nn_preds_val[-1].reshape(-1), nn_preds_test[:-1].reshape(-1)])
    y_true = np.concatenate([y_val[-1].reshape(-1), y_test[:-1].reshape(-1)])
    lin_errors = (last_preds_lin-y_true)**2
    nn_errors = (last_preds_nn-y_true)**2

    return lin_errors, nn_errors

def oracle(lin_preds, nn_preds, y_test, p, thresh=1e-3):
    y_test = y_test.reshape(-1)
    lin_preds = lin_preds.reshape(-1)
    nn_preds = nn_preds.reshape(-1)

    lin_errors = (lin_preds-y_test)**2
    nn_errors = (nn_preds-y_test)**2
    
    # How many to pick
    selection = np.zeros((len(y_test)))
    n_lin = int(np.ceil(p * len(y_test)))

    loss_diff = (lin_errors - nn_errors)
    I = np.argsort(loss_diff)
    if thresh is not None:
        try:
            n_max = np.where(np.sort(loss_diff) > thresh)[0][0]
        except IndexError:
            n_max = len(loss_diff)
        if n_max >= n_lin:
            selection[I[:n_max]] = 1
            return selection.astype(np.int8)
    
    selection[I[:n_lin]] = 1
    return selection.astype(np.int8)

# Select linear model p_lin percent of the time. Choose the 1-p_lin percent worst predictions and reduce via neural net prediction
def selection_oracle_percent(y_test, lin_test_preds, ens_test_preds, p_lin):
    n_test = len(y_test)
    selection = np.ones((n_test))
    se_lin = (lin_test_preds.squeeze()-y_test.squeeze())**2
    se_ens = (ens_test_preds.squeeze()-y_test.squeeze())**2

    ens_better = np.where(se_ens < se_lin)[0]
    loss_diff = (se_lin[ens_better] - se_ens[ens_better])**2

    # How many datapoints are (1-p_lin) percent?
    n_ens = int(n_test * (1-p_lin))
    
    # Substitute worst offenders
    substitue_indices = ens_better[np.argsort(-loss_diff)[:n_ens]]
    selection[substitue_indices] = 0
    return selection.astype(np.int8)

def get_roc_dists(x_test, lin_rocs, ensemble_rocs, distance='euclidean'):

    if distance == 'euclidean':
        lin_min_dist = np.min(np.vstack([lin_roc.euclidean_distance(x_test) for lin_roc in lin_rocs]), axis=0)
        ensemble_min_dist = np.min(np.vstack([ensemble_roc.euclidean_distance(x_test) for ensemble_roc in ensemble_rocs]), axis=0)
    else:
        lin_min_dist = np.min(np.vstack([lin_roc.dtw_distance(x_test) for lin_roc in lin_rocs]), axis=0)
        ensemble_min_dist = np.min(np.vstack([ensemble_roc.dtw_distance(x_test) for ensemble_roc in ensemble_rocs]), axis=0)

    return lin_min_dist, ensemble_min_dist

def run_v12(lin_train_preds, ens_train_preds, y_train, y_test, x_val, y_val, x_test, lin_val_preds, ens_val_preds, lin_test_preds, ens_test_preds, random_state, p=0.9, return_predictor=False):
    name = f'v12_{p}'

    # Get oracle prediction
    val_selection = oracle(lin_val_preds, ens_val_preds, y_val,  p=p)
    n_zeros = (val_selection == 0).sum()
    if n_zeros == 0:
        return name, np.ones((len(x_test))).astype(np.int8), None
    if n_zeros == len(val_selection):
        return name, np.zeros((len(x_test))).astype(np.int8), None

    # Last errors
    lin_errors, nn_errors = get_last_errors(lin_train_preds, ens_train_preds, y_train, lin_val_preds, ens_val_preds, y_val)
    train_last_errors = (lin_errors - nn_errors)

    lin_errors, nn_errors = get_last_errors(lin_val_preds, ens_val_preds, y_val, lin_test_preds, ens_test_preds, y_test)
    test_last_errors = (lin_errors - nn_errors)

    # Build X
    X_train = np.vstack([lin_val_preds-ens_val_preds, train_last_errors]).T
    X_train = np.concatenate([X_train, x_val], axis=1)

    X_test = np.vstack([lin_test_preds-ens_test_preds, test_last_errors]).T
    X_test = np.concatenate([X_test, x_test], axis=1)

    # Train model(s)
    clf = UpsampleEnsembleClassifier(RandomForestClassifier, 9, random_state=random_state, n_estimators=128)
    clf.fit(X_train, val_selection)
    gfi = clf.global_feature_importance()

    thresh = 0.6 if p > 0.5 else 0.5
    #thresh = 0.5

    if return_predictor:
        return clf, thresh

    return name, clf.predict(X_test, thresh=thresh).astype(np.int8), gfi