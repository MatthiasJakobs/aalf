import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from imblearn.ensemble import BalancedRandomForestClassifier

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


def get_roc_dists(x_test, lin_rocs, ensemble_rocs):

    lin_min_dist = np.min(np.vstack([lin_roc.euclidean_distance(x_test) for lin_roc in lin_rocs]), axis=0)
    ensemble_min_dist = np.min(np.vstack([ensemble_roc.euclidean_distance(x_test) for ensemble_roc in ensemble_rocs]), axis=0)

    return lin_min_dist, ensemble_min_dist

'''
    Compare Regions of Competence. Choose linear if RoC distance is smaller
'''
def run_v1(x_test, lin_rocs, ensemble_rocs):
    test_selection = []
    name = 'v1'

    # What to do if empty rocs
    if len(ensemble_rocs) == 0:
        # Choose linear
        return name, np.ones((len(x_test))).astype(np.int8)
    if len(lin_rocs) == 0:
        # Choose complex
        return name, np.zeros((len(x_test))).astype(np.int8)

    lin_min_dist, ensemble_min_dist = get_roc_dists(x_test, lin_rocs, ensemble_rocs)
    test_selection = (lin_min_dist <= ensemble_min_dist).astype(np.int8)

    return name, test_selection

'''
    Compare Regions of Competence. Choose linear if RoC distance is smaller.
    Otherwise, roll the dice and still choose linear model with probability p_l
'''
def run_v2(x_test, lin_rocs, ensemble_rocs, p_l, random_state=None):
    test_selection = []
    name = f'v2_{p_l}'

    # What to do if empty rocs
    if len(ensemble_rocs) == 0:
        # Choose linear
        return name, np.ones((len(x_test))).astype(np.int8)
    if len(lin_rocs) == 0:
        # Choose complex
        return name, np.zeros((len(x_test))).astype(np.int8)

    rng = np.random.RandomState(random_state)
    lin_min_dist, ensemble_min_dist = get_roc_dists(x_test, lin_rocs, ensemble_rocs)
    test_selection = (lin_min_dist <= ensemble_min_dist).astype(np.int8)

    # Still choose linear with probability p_l
    lin_worse = np.where(test_selection == 0)[0]
    test_selection[lin_worse] = rng.binomial(1, p_l, size=len(lin_worse))

    return name, test_selection

# If the ratio is not too large, still choose linear model
def run_v3(x_test, lin_rocs, ensemble_rocs, thresh):
    test_selection = []
    name = f'v3_{thresh}'

    # What to do if empty rocs
    if len(ensemble_rocs) == 0:
        # Choose linear
        return name, np.ones((len(x_test))).astype(np.int8)
    if len(lin_rocs) == 0:
        # Choose complex
        return name, np.zeros((len(x_test))).astype(np.int8)

    lin_min_dist, ensemble_min_dist = get_roc_dists(x_test, lin_rocs, ensemble_rocs)
    test_selection = (lin_min_dist <= ensemble_min_dist).astype(np.int8)
    lin_worse = np.where(np.bitwise_and(test_selection == 0, ensemble_min_dist > 0))[0]

    ratios = lin_min_dist[lin_worse] / ensemble_min_dist[lin_worse]
    test_selection[lin_worse] = (ratios <= thresh).astype(np.int8)

    return name, test_selection

# Train classifier based on RoC distances and optimal validation selection
def run_v4(val_selection, x_val, x_test, lin_val_preds, ens_val_preds, lin_test_preds, ens_test_preds, lin_rocs, ensemble_rocs, random_state, calibrate=False, thresh=0.5, use_rocs=True):
    if calibrate:
        name = f'v4_{thresh}_calibrated'
    else:
        name = f'v4_{thresh}'
    if not use_rocs:
        name = name + '_norocs'
    # What to do if empty rocs
    if len(ensemble_rocs) == 0:
        # Choose linear
        return name, np.ones((len(x_test))).astype(np.int8)
    if len(lin_rocs) == 0:
        # Choose complex
        return name, np.zeros((len(x_test))).astype(np.int8)

    lin_min_dist, ensemble_min_dist = get_roc_dists(x_val, lin_rocs, ensemble_rocs)

    if use_rocs:
        X_train = np.vstack([lin_val_preds, ens_val_preds, lin_min_dist, ensemble_min_dist]).T
    else:
        X_train = np.vstack([lin_val_preds, ens_val_preds]).T
    #X_train = np.concatenate([X_train, x_val], axis=1)
    y_train = val_selection

    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_train, y_train)

    lin_min_dist, ensemble_min_dist = get_roc_dists(x_test, lin_rocs, ensemble_rocs)
    if use_rocs:
        X_test = np.vstack([lin_test_preds, ens_test_preds, lin_min_dist, ensemble_min_dist]).T
    else:
        X_test = np.vstack([lin_test_preds, ens_test_preds]).T
    #X_test = np.concatenate([X_test, x_test], axis=1)

    # Very simple calibration 
    best_thresh = thresh
    if calibrate:
        n_calib = 15
        fn_train = len(y_train) - np.sum(clf.predict(X_train)[y_train == 1])
        thresholds = np.arange(n_calib)[1:] / (2*n_calib)
        for _thresh in thresholds[::-1]:
            new_fn = len(y_train) - np.sum((clf.predict_proba(X_train)[:, 1] > _thresh)[y_train == 1])
            if new_fn < fn_train:
                fn_train = new_fn
                best_thresh = _thresh
        # #print('best thresh', best_thresh, min(thresholds), max(thresholds))
    
    return name, (clf.predict_proba(X_test)[:, 1] > best_thresh).astype(np.int8)

# Train on biggest errors in validation set
def run_v5(x_val, y_val, x_test, y_test, lin_val_preds, ens_val_preds, lin_test_preds, ens_test_preds, lin_rocs, ensemble_rocs, random_state, p=0.99, calibrate=False):
    if calibrate:
        name = f'v5_{p}_calibrated'
    else:
        name = f'v5_{p}'
    # What to do if empty rocs
    if len(ensemble_rocs) == 0:
        # Choose linear
        return name, np.ones((len(x_test))).astype(np.int8)
    if len(lin_rocs) == 0:
        # Choose complex
        return name, np.zeros((len(x_test))).astype(np.int8)

    lin_min_dist, ensemble_min_dist = get_roc_dists(x_val, lin_rocs, ensemble_rocs)

    y_train = selection_oracle_percent(y_val, lin_val_preds, ens_val_preds, p)

    # lin_min_dist_indices = np.argmin(np.vstack([lin_roc.euclidean_distance(x_val) for lin_roc in lin_rocs]), axis=0)
    # ensemble_min_dist_indices = np.argmin(np.vstack([ensemble_roc.euclidean_distance(x_val) for ensemble_roc in ensemble_rocs]), axis=0)
    # se_lin = [lin_rocs[_idx] for _idx in lin_min_dist_indices]
    # se_ens = [ensemble_rocs[_idx] for _idx in ensemble_min_dist_indices]
    # lin_closest_roc_losses = np.array([roc.squared_error for roc in se_lin])
    # nn_closest_roc_losses = np.array([roc.squared_error for roc in se_ens])

    # X_train = np.vstack([lin_closest_roc_losses / nn_closest_roc_losses, lin_val_preds, ens_val_preds, lin_min_dist, ensemble_min_dist]).T

    #X_train = np.vstack([lin_val_preds, ens_val_preds, lin_min_dist, ensemble_min_dist]).T
    X_train = np.vstack([(lin_val_preds-ens_val_preds)**2, (lin_min_dist-ensemble_min_dist)**2]).T
    X_train = np.concatenate([X_train, x_val], axis=1)

    clf = BalancedRandomForestClassifier(n_estimators=128, replacement=True, random_state=random_state, sampling_strategy='not minority')
    #clf = RandomForestClassifier(n_estimators=128, random_state=random_state)
    clf.fit(X_train, y_train)

    lin_min_dist, ensemble_min_dist = get_roc_dists(x_test, lin_rocs, ensemble_rocs)

    # lin_min_dist_indices = np.argmin(np.vstack([lin_roc.euclidean_distance(x_test) for lin_roc in lin_rocs]), axis=0)
    # ensemble_min_dist_indices = np.argmin(np.vstack([ensemble_roc.euclidean_distance(x_test) for ensemble_roc in ensemble_rocs]), axis=0)
    # se_lin = [lin_rocs[_idx] for _idx in lin_min_dist_indices]
    # se_ens = [ensemble_rocs[_idx] for _idx in ensemble_min_dist_indices]
    # lin_closest_roc_losses = np.array([roc.squared_error for roc in se_lin])
    # nn_closest_roc_losses = np.array([roc.squared_error for roc in se_ens])
    # X_test = np.vstack([lin_closest_roc_losses / nn_closest_roc_losses, lin_test_preds, ens_test_preds, lin_min_dist, ensemble_min_dist]).T

    #X_test = np.vstack([lin_test_preds, ens_test_preds, lin_min_dist, ensemble_min_dist]).T
    X_test = np.vstack([(lin_test_preds-ens_test_preds)**2, (lin_min_dist-ensemble_min_dist)**2]).T
    X_test = np.concatenate([X_test, x_test], axis=1)
    oracle_y = selection_oracle_percent(y_test, lin_test_preds, ens_test_preds, p)
    #print(p, 'train f1', f'{f1_score(clf.predict(X_train), y_train):.3f}', 'test f1', f'{f1_score(clf.predict(X_test), oracle_y):.3f}', f'{clf.predict(X_test).mean():.3f}')


    return name, clf.predict(X_test)

# Idea: Take loss achieved for RoCMember as proxy for expected loss on new data
def run_v6(x_test, lin_rocs, nn_rocs):
    name = 'v6'
    selection = []

    # What to do if empty rocs
    if len(nn_rocs) == 0:
        # Choose linear
        return name, np.ones((len(x_test))).astype(np.int8)
    if len(lin_rocs) == 0:
        # Choose complex
        return name, np.zeros((len(x_test))).astype(np.int8)

    for x in x_test:
        lin_min_dist_idx = np.argmin([lin_roc.euclidean_distance(x) for lin_roc in lin_rocs])
        nn_min_dist_idx = np.argmin([nn_roc.euclidean_distance(x) for nn_roc in nn_rocs])

        lin_closest_roc = lin_rocs[lin_min_dist_idx]
        nn_closest_roc = nn_rocs[nn_min_dist_idx]

        selection.append(lin_closest_roc.squared_error <= nn_closest_roc.squared_error)

    return name, np.array(selection).astype(np.int8)

# optimize p_t based on formula with neural network
def run_v7(x_val, x_test, lin_rocs, lin_val_preds, ensemble_rocs, ensemble_val_preds):
    ### Construct meta dataset
    # Distance to closest RoC member
    lin_min_dist, ensemble_min_dist = get_roc_dists(x_val, lin_rocs, ensemble_rocs)

    # Expected losses based on saved RoC member
    lin_min_dist_indices = np.argmin(np.vstack([lin_roc.euclidean_distance(x_val) for lin_roc in lin_rocs]), axis=0)
    ensemble_min_dist_indices = np.argmin(np.vstack([ensemble_roc.euclidean_distance(x_val) for ensemble_roc in ensemble_rocs]), axis=0)
    se_lin = [lin_rocs[_idx] for _idx in lin_min_dist_indices]
    se_ens = [ensemble_rocs[_idx] for _idx in ensemble_min_dist_indices]
    lin_closest_roc_losses = np.array([roc.squared_error for roc in se_lin])
    nn_closest_roc_losses = np.array([roc.squared_error for roc in se_ens])

    X_train = np.vstack([lin_min_dist / ensemble_min_dist, lin_closest_roc_losses / nn_closest_roc_losses, ]).T
    y_train = np.zeros((len(X_train)))


    exit()