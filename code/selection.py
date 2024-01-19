import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from imblearn.ensemble import BalancedRandomForestClassifier
from models import DownsampleEnsembleClassifier, UpsampleEnsembleClassifier

def get_last_errors(lin_preds_val, nn_preds_val, y_val, lin_preds_test, nn_preds_test, y_test):
    last_preds_lin = np.concatenate([lin_preds_val[-1].reshape(-1), lin_preds_test[:-1].reshape(-1)])
    last_preds_nn = np.concatenate([nn_preds_val[-1].reshape(-1), nn_preds_test[:-1].reshape(-1)])
    y_true = np.concatenate([y_val[-1].reshape(-1), y_test[:-1].reshape(-1)])
    lin_errors = (last_preds_lin-y_true)**2
    nn_errors = (last_preds_nn-y_true)**2

    return lin_errors, nn_errors

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
def run_v4(val_selection, x_val, x_test, lin_val_preds, ens_val_preds, lin_test_preds, ens_test_preds, lin_rocs, ensemble_rocs, random_state, calibrate=False, thresh=0.5, use_diff=False):
    if calibrate:
        name = f'v4_{thresh}_calibrated'
    else:
        name = f'v4_{thresh}'
    if use_diff:
        name = name + '_diff'
    # What to do if empty rocs
    if len(ensemble_rocs) == 0:
        # Choose linear
        return name, np.ones((len(x_test))).astype(np.int8)
    if len(lin_rocs) == 0:
        # Choose complex
        return name, np.zeros((len(x_test))).astype(np.int8)

    lin_min_dist, ensemble_min_dist = get_roc_dists(x_val, lin_rocs, ensemble_rocs)

    X_train = np.vstack([lin_val_preds, ens_val_preds, lin_min_dist, ensemble_min_dist]).T
    if use_diff:
        X_train = np.vstack([lin_val_preds-ens_val_preds, lin_min_dist-ensemble_min_dist]).T
    #X_train = np.concatenate([X_train, x_val], axis=1)
    y_train = val_selection

    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_train, y_train)

    lin_min_dist, ensemble_min_dist = get_roc_dists(x_test, lin_rocs, ensemble_rocs)
    X_test = np.vstack([lin_test_preds, ens_test_preds, lin_min_dist, ensemble_min_dist]).T
    if use_diff:
        X_test = np.vstack([lin_test_preds-ens_test_preds, lin_min_dist-ensemble_min_dist]).T
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
def run_v5(x_val, y_val, x_test, lin_val_preds, ens_val_preds, lin_test_preds, ens_test_preds, lin_rocs, ensemble_rocs, random_state):
    name = f'v5'
    # What to do if empty rocs
    if len(ensemble_rocs) == 0:
        # Choose linear
        return name, np.ones((len(x_test))).astype(np.int8)
    if len(lin_rocs) == 0:
        # Choose complex
        return name, np.zeros((len(x_test))).astype(np.int8)

    lin_min_dist, ensemble_min_dist = get_roc_dists(x_val, lin_rocs, ensemble_rocs)

    y_train = selection_oracle_percent(y_val, lin_val_preds, ens_val_preds, 0.5)

    X_train = np.vstack([lin_val_preds, ens_val_preds, lin_min_dist, ensemble_min_dist]).T
    #X_train = np.concatenate([X_train, x_val], axis=1)

    clf = BalancedRandomForestClassifier(n_estimators=128, replacement=True, random_state=random_state, sampling_strategy='not minority')
    #clf = RandomForestClassifier(n_estimators=128, random_state=random_state)
    clf.fit(X_train, y_train)

    lin_min_dist, ensemble_min_dist = get_roc_dists(x_test, lin_rocs, ensemble_rocs)

    X_test = np.vstack([lin_test_preds, ens_test_preds, lin_min_dist, ensemble_min_dist]).T

    return name, (clf.predict_proba(X_test)[:, 1] > 0.5).astype(np.int8)

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
def run_v7(x_test, lin_rocs, ensemble_rocs, lin_test_preds, ensemble_test_preds):
    name = 'v7'

    # What to do if empty rocs
    if len(ensemble_rocs) == 0:
        # Choose linear
        return name, np.ones((len(x_test))).astype(np.int8)
    if len(lin_rocs) == 0:
        # Choose complex
        return name, np.zeros((len(x_test))).astype(np.int8)

    lin_min_dist, ensemble_min_dist = get_roc_dists(x_test, lin_rocs, ensemble_rocs)

    from test_nn import Net
    import torch
    model = Net()
    model.load_state_dict(torch.load('model.net', map_location='cpu'))

    X = np.vstack([lin_test_preds, ensemble_test_preds, lin_min_dist, ensemble_min_dist]).T
    return name, model.predict(X)

def run_v8(x_val, y_val, x_test, lin_val_preds, ens_val_preds, lin_test_preds, ens_test_preds, lin_rocs, ensemble_rocs, random_state, thresh=0.5):
    name = 'v8'

    # What to do if empty rocs
    if len(ensemble_rocs) == 0:
        # Choose linear
        return name, np.ones((len(x_test))).astype(np.int8)
    if len(lin_rocs) == 0:
        # Choose complex
        return name, np.zeros((len(x_test))).astype(np.int8)

    lin_min_dist, ensemble_min_dist = get_roc_dists(x_val, lin_rocs, ensemble_rocs)

    y_train = selection_oracle_percent(y_val, lin_val_preds, ens_val_preds, 0.5)

    X_train = np.vstack([lin_val_preds-ens_val_preds, lin_min_dist-ensemble_min_dist]).T
    X_train = np.concatenate([X_train, x_val], axis=1)

    clf = BalancedRandomForestClassifier(n_estimators=256, replacement=True, random_state=random_state, sampling_strategy='not minority')
    #clf = RandomForestClassifier(n_estimators=128, random_state=random_state)
    clf.fit(X_train, y_train)

    lin_min_dist, ensemble_min_dist = get_roc_dists(x_test, lin_rocs, ensemble_rocs)

    X_test = np.vstack([lin_test_preds-ens_test_preds, lin_min_dist-ensemble_min_dist]).T
    X_test = np.concatenate([X_test, x_test], axis=1)

    return name, (clf.predict_proba(X_test)[:, 1] > thresh).astype(np.int8)

def run_v9(x_val, y_val, x_test, lin_val_preds, ens_val_preds, lin_test_preds, ens_test_preds, lin_rocs, ensemble_rocs, random_state, thresh=0.5):
    name = 'v9'

    # What to do if empty rocs
    if len(ensemble_rocs) == 0:
        # Choose linear
        return name, np.ones((len(x_test))).astype(np.int8)
    if len(lin_rocs) == 0:
        # Choose complex
        return name, np.zeros((len(x_test))).astype(np.int8)

    lin_min_dist, ensemble_min_dist = get_roc_dists(x_val, lin_rocs, ensemble_rocs)

    y_train = selection_oracle_percent(y_val, lin_val_preds, ens_val_preds, 0.5)

    X_train = np.vstack([lin_val_preds-ens_val_preds]).T
    X_train = np.concatenate([X_train, x_val], axis=1)

    clf = BalancedRandomForestClassifier(n_estimators=256, replacement=True, random_state=random_state, sampling_strategy='not minority')
    #clf = DecisionTreeClassifier(max_depth=4, random_state=random_state)
    clf.fit(X_train, y_train)

    lin_min_dist, ensemble_min_dist = get_roc_dists(x_test, lin_rocs, ensemble_rocs)

    X_test = np.vstack([lin_test_preds-ens_test_preds]).T
    X_test = np.concatenate([X_test, x_test], axis=1)

    return name, (clf.predict_proba(X_test)[:, 1] > thresh).astype(np.int8)

# def run_test(x_val, y_val, x_test, lin_val_preds, ens_val_preds, lin_test_preds, ens_test_preds, lin_rocs, ensemble_rocs, random_state, thresh=0.5):
#     name = 'test'

#     # What to do if empty rocs
#     if len(ensemble_rocs) == 0:
#         # Choose linear
#         return name, np.ones((len(x_test))).astype(np.int8)
#     if len(lin_rocs) == 0:
#         # Choose complex
#         return name, np.zeros((len(x_test))).astype(np.int8)

#     lin_min_dist, ensemble_min_dist = get_roc_dists(x_val, lin_rocs, ensemble_rocs)

#     y_train = selection_oracle_percent(y_val, lin_val_preds, ens_val_preds, 0.5)

#     X_train = np.vstack([lin_val_preds-ens_val_preds]).T
#     X_train = np.concatenate([X_train, x_val], axis=1)

#     clf = BalancedRandomForestClassifier(n_estimators=128, replacement=True, random_state=random_state, sampling_strategy='not minority')
#     #clf = DecisionTreeClassifier(max_depth=4, random_state=random_state)
#     clf.fit(X_train, y_train)

#     lin_min_dist, ensemble_min_dist = get_roc_dists(x_test, lin_rocs, ensemble_rocs)

#     X_test = np.vstack([lin_test_preds-ens_test_preds]).T
#     X_test = np.concatenate([X_test, x_test], axis=1)

#     return name, (clf.predict_proba(X_test)[:, 1] > thresh).astype(np.int8)

# last error approx equal expected new error
def run_test(y_val, y_test, lin_preds_val, nn_preds_val, lin_preds_test, nn_preds_test, epsilon=1):
    # First error estimate from validation set
    e_i = (lin_preds_val[-1]-y_val[-1])**2
    e_c = (nn_preds_val[-1]-y_val[-1])**2

    name = 'test'
    selection = []
    for t in range(len(y_test)):
        if e_i == 0:
            selection.append(1)
        elif e_c == 0:
            selection.append(0)
        else:
            if e_i / e_c <= epsilon:
                selection.append(1)
            else:
                selection.append(0)
        
        # Decision is made, update error estimate
        e_i = (lin_preds_test[t]-y_test[t])**2
        e_c = (nn_preds_test[t]-y_test[t])**2

    return name, np.array(selection).astype(np.int8)

def run_v10(y_train, x_val, y_val, x_test, y_test, lin_train_preds, ens_train_preds, lin_val_preds, ens_val_preds, lin_test_preds, ens_test_preds, lin_rocs, ensemble_rocs, random_state, thresh=0.5):

    name = 'v10'
    # What to do if empty rocs
    if len(ensemble_rocs) == 0:
        # Choose linear
        return name, np.ones((len(x_test))).astype(np.int8)
    if len(lin_rocs) == 0:
        # Choose complex
        return name, np.zeros((len(x_test))).astype(np.int8)


    # Get oracle prediction
    val_selection = selection_oracle_percent(y_val, lin_val_preds, ens_val_preds, 0.9)

    # Build X
    lin_min_dist, ens_min_dist = get_roc_dists(x_val, lin_rocs, ensemble_rocs)
    X_train = np.vstack([lin_val_preds-ens_val_preds, lin_min_dist-ens_min_dist]).T
    X_train = np.concatenate([X_train, x_val], axis=1)

    lin_min_dist, ens_min_dist = get_roc_dists(x_test, lin_rocs, ensemble_rocs)
    X_test = np.vstack([lin_test_preds-ens_test_preds, lin_min_dist-ens_min_dist]).T
    X_test = np.concatenate([X_test, x_test], axis=1)

    # Last errors
    lin_errors, nn_errors = get_last_errors(lin_train_preds, ens_train_preds, y_train, lin_val_preds, ens_val_preds, y_val)
    train_last_errors = (lin_errors / (nn_errors+1e-4)).reshape(-1, 1)

    lin_errors, nn_errors = get_last_errors(lin_val_preds, ens_val_preds, y_val, lin_test_preds, ens_test_preds, y_test)
    test_last_errors = (lin_errors / (nn_errors+1e-4)).reshape(-1, 1)

    # X_train = np.concatenate([X_train, train_last_errors], axis=1)
    # X_test = np.concatenate([X_test, test_last_errors], axis=1)

    # Train model(s)
    '''
    rf_rng = np.random.RandomState(random_state)
    one_indices = np.where(val_selection == 1)[0]
    zero_indices = np.where(val_selection == 0)[0]

    # Upsample minority
    indices = rf_rng.choice(zero_indices, size=len(one_indices), replace=True)
    indices = np.concatenate([indices, one_indices])
    _x = X_train[indices]
    _y = val_selection[indices]

    clf = RandomForestClassifier(n_estimators=128, random_state=rf_rng)
    clf.fit(_x, _y)
    '''
    clf = DownsampleEnsembleClassifier(RandomForestClassifier, 10, random_state=random_state, n_estimators=128)
    clf.fit(X_train, val_selection)

    return name, clf.predict(X_test).astype(np.int8)

def run_v11(x_val, y_val, x_test, lin_val_preds, ens_val_preds, lin_test_preds, ens_test_preds, random_state, p=0.9):

    name = f'v11_{p}'

    # Get oracle prediction
    val_selection = selection_oracle_percent(y_val, lin_val_preds, ens_val_preds, p)
    n_zeros = (val_selection == 0).sum()
    if n_zeros == 0:
        return name, np.ones((len(x_test))).astype(np.int8)
    if n_zeros == len(val_selection):
        return name, np.zeros((len(x_test))).astype(np.int8)

    # Build X
    X_train = np.vstack([lin_val_preds-ens_val_preds]).T
    X_train = np.concatenate([X_train, x_val], axis=1)

    X_test = np.vstack([lin_test_preds-ens_test_preds]).T
    X_test = np.concatenate([X_test, x_test], axis=1)

    # Train model(s)
    clf = UpsampleEnsembleClassifier(RandomForestClassifier, 9, random_state=random_state, n_estimators=128)
    clf.fit(X_train, val_selection)

    return name, clf.predict(X_test, thresh=0.6).astype(np.int8)

def run_sebas_selection(y_test, lin_test_preds, nn_test_preds, n_i, n_c):
    # Naive, slow version
    name = 'sebas01'
    selection = []

    e_i = (y_test.reshape(-1) - lin_test_preds.reshape(-1))**2
    e_c = (y_test.reshape(-1) - nn_test_preds.reshape(-1))**2

    run_lin = True
    t = 0
    while t < len(y_test):
        if run_lin:
            selection.extend(n_i * [1])
            t += n_i 
            run_lin = False
            continue
        if np.mean(e_c[t-n_i:t]) < np.mean(e_i[t-n_i:t]):
            selection.extend(n_c*[0])
            t += n_c
            run_lin = True
            continue
        else:
            selection.append(1)
            t += 1
            continue

    return name, np.array(selection)[:len(y_test)].astype(np.int8)



