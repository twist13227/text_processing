import numpy as np


def kfold_split(num_objects, num_folds):
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects (int): number of objects in train set
    num_folds (int): number of folds for cross-validation split

    Returns:
    list((tuple(np.array, np.array))): list of length num_folds, where i-th element of list contains tuple of 2 numpy arrays,
                                       the 1st numpy array contains all indexes without i-th fold while the 2nd one contains
                                       i-th fold
    """
    fold_size = num_objects // num_folds
    ind_array = np.array([i for i in range(num_objects)])
    res = []
    for i in range(num_folds - 1):
        res.append(
            (
                np.hstack(
                    (ind_array[: fold_size * i], ind_array[fold_size * (i + 1) :])
                ),
                ind_array[fold_size * i : fold_size * (i + 1)],
            )
        )
    res.append(
        (
            ind_array[: fold_size * (num_folds - 1)],
            ind_array[fold_size * (num_folds - 1) :],
        )
    )
    return res


def knn_cv_score(X, y, parameters, score_function, folds, knn_class):
    """Takes train data, counts cross-validation score over grid of parameters (all possible parameters combinations)

    Parameters:
    X (2d np.array): train set
    y (1d np.array): train labels
    parameters (dict): dict with keys from {n_neighbors, metrics, weights, normalizers}, values of type list,
                       parameters['normalizers'] contains tuples (normalizer, normalizer_name), see parameters
                       example in your jupyter notebook
    score_function (callable): function with input (y_true, y_predict) which outputs score metric
    folds (list): output of kfold_split
    knn_class (obj): class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight), value - mean score over all folds
    """
    res = dict()
    for n_neighbors in parameters["n_neighbors"]:
        for metric in parameters["metrics"]:
            for weights in parameters["weights"]:
                for normalizer in parameters["normalizers"]:
                    scores = []
                    for train_test in folds:
                        X_train = X[train_test[0]]
                        y_train = y[train_test[0]]
                        X_test = X[train_test[1]]
                        y_test = y[train_test[1]]
                        if normalizer[1] != "None":
                            normalizer[0].fit(X_train)
                            X_train = normalizer[0].transform(X_train)
                            X_test = normalizer[0].transform(X_test)
                        neigh = knn_class(
                            n_neighbors=n_neighbors, weights=weights, metric=metric
                        )
                        neigh.fit(X_train, y_train)
                        y_pred = neigh.predict(X_test)
                        scores.append(score_function(y_test, y_pred, average="macro"))
                        score = sum(scores) / len(scores)
                        res[(normalizer[1], n_neighbors, metric, weights)] = score
    return res
