import numpy as np

'''
    Compare Regions of Competence. Choose linear if RoC distance is smaller
'''
def run_v1(x_test, lin_rocs, ensemble_rocs):
    test_selection = []
    name = 'v1'
    for _x in x_test:
        
        # What to do if empty rocs
        if len(ensemble_rocs) == 0:
            # Choose linear
            test_selection.append(1)
            continue
        if len(lin_rocs) == 0:
            test_selection.append(0)
            continue

        lin_min_dist = min([lin_roc.euclidean_distance(_x) for lin_roc in lin_rocs])
        ensemble_min_dist = min([ensemble_roc.euclidean_distance(_x) for ensemble_roc in ensemble_rocs])
        if lin_min_dist <= ensemble_min_dist:
            test_selection.append(1)
        else:
            test_selection.append(0)

    test_selection = np.array(test_selection)
    return name, test_selection

'''
    Compare Regions of Competence. Choose linear if RoC distance is smaller.
    Otherwise, roll the dice and still choose linear model with probability p_l
'''
def run_v2(x_test, lin_rocs, ensemble_rocs, p_l, random_state=None):
    test_selection = []
    name = f'v2_{p_l}'
    rng = np.random.RandomState(random_state)
    for _x in x_test:
        
        # What to do if empty rocs
        if len(ensemble_rocs) == 0:
            # Choose linear
            test_selection.append(1)
            continue
        if len(lin_rocs) == 0:
            test_selection.append(0)
            continue

        lin_min_dist = min([lin_roc.euclidean_distance(_x) for lin_roc in lin_rocs])
        ensemble_min_dist = min([ensemble_roc.euclidean_distance(_x) for ensemble_roc in ensemble_rocs])
        if lin_min_dist <= ensemble_min_dist:
            test_selection.append(1)
        else:
            # Still choose linear with probability p_l
            test_selection.append(rng.binomial(1, p_l))

    test_selection = np.array(test_selection)
    return name, test_selection

# If the ratio is not too large, still choose linear model
def run_v3(x_test, lin_rocs, ensemble_rocs, thresh):
    test_selection = []
    name = f'v3_{thresh}'
    for _x in x_test:
        
        # What to do if empty rocs
        if len(ensemble_rocs) == 0:
            # Choose linear
            test_selection.append(1)
            continue
        if len(lin_rocs) == 0:
            test_selection.append(0)
            continue

        lin_min_dist = min([lin_roc.euclidean_distance(_x) for lin_roc in lin_rocs])
        ensemble_min_dist = min([ensemble_roc.euclidean_distance(_x) for ensemble_roc in ensemble_rocs])

        if lin_min_dist <= ensemble_min_dist:
            test_selection.append(1)
        else:
            ratio = lin_min_dist / ensemble_min_dist
            if ratio <= thresh:
                test_selection.append(1)
            else:
                test_selection.append(0)

    test_selection = np.array(test_selection)
    return name, test_selection