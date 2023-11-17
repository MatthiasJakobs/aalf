import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

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
def run_v4(val_selection, x_val, x_test, lin_val_preds, ens_val_preds, lin_test_preds, ens_test_preds, lin_rocs, ensemble_rocs, random_state, include_predictions=True):
    if include_predictions:
        name = f'v4_both'
    else:
        name = f'v4_roc'
    # What to do if empty rocs
    if len(ensemble_rocs) == 0:
        # Choose linear
        return name, np.ones((len(x_test))).astype(np.int8)
    if len(lin_rocs) == 0:
        # Choose complex
        return name, np.zeros((len(x_test))).astype(np.int8)

    lin_min_dist, ensemble_min_dist = get_roc_dists(x_val, lin_rocs, ensemble_rocs)

    if include_predictions:
        X_train = np.vstack([lin_val_preds, ens_val_preds, lin_min_dist, ensemble_min_dist]).T
    else:
        X_train = np.vstack([lin_min_dist, ensemble_min_dist]).T
    y_train = val_selection

    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_train, y_train)

    lin_min_dist, ensemble_min_dist = get_roc_dists(x_test, lin_rocs, ensemble_rocs)
    if include_predictions:
        X_test = np.vstack([lin_test_preds, ens_test_preds, lin_min_dist, ensemble_min_dist]).T
    else:
        X_test = np.vstack([lin_min_dist, ensemble_min_dist]).T

    return name, clf.predict(X_test)
