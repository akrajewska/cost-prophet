import functools

import dask
import numpy as np
import pandas as pd
from fancyimpute import Solver
from dask.distributed import Client
from cost_prophet.utils.evaluation import test_error
from dask import delayed, compute


def split_tests_sets(known_indices: list, X: np.ndarray, split_rate: float = 0.75) -> tuple:
    test_indices = []
    train_indices = []
    X_train = np.array(X, copy=True)
    for idx in known_indices:
        ro = np.random.uniform()
        if ro < split_rate and np.count_nonzero(~np.isnan(X_train[:, idx[1]])) > 1 and np.count_nonzero(
                ~np.isnan(X_train[idx[0]])) > 1:
            test_indices.append(idx)
            X_train[tuple(idx)] = np.nan
        else:
            train_indices.append(idx)
    return test_indices, train_indices, X_train


def get_known_indices(X: np.ndarray) -> list:
    return np.argwhere(~np.isnan(X)).tolist()


def run_trial(solver, param, X, known_indices, trial):
    test_indices, train_indices, X_train = split_tests_sets(known_indices, X)
    X_soft_impute_results = solver.run(X_train, trial)
    results = []
    for shrinkage_value, X_out in X_soft_impute_results:
        test_error = test_error(X_out, X, test_indices)
        results.append([param, trial, shrinkage_value, test_error])
    return results

def experiment(solver, trials, X):
    errors = []
    known_indices = get_known_indices(X)
    params = [1]
    for param in params:
        for trial in range(trials):
            _nrmse_data = delayed(run_trial)(solver, param, X, known_indices, trial)
            errors.append(_nrmse_data)
    errors = compute(errors)
    #TODO dlaczego jest tuple?
    df = pd.DataFrame(data=errors[0], columns=['param', 'trial', 'shrinkage value', 'test error'])
    return df
