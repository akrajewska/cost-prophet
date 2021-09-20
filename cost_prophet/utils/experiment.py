import os

import pandas as pd
from dask import delayed, compute
from dotenv import dotenv_values

from cost_prophet.utils.evaluation import test_error
from cost_prophet.utils.linear_alebra import get_known_indices, split_tests_sets

config = dotenv_values()

OUTPUT_DIR = config.get("OUTPUT_DIR")


def run_trial(solver, X, known_indices, trial):
    test_indices, train_indices, X_train = split_tests_sets(known_indices, X)
    X_soft_impute_results = solver.fit_transform(X_train)
    results = []
    for shrinkage_value, X_out in X_soft_impute_results:
        _test_error = test_error(X_out, X, test_indices)
        results.append([trial, shrinkage_value, _test_error])
    return results


def experiment(solver, trials, X):
    errors = []
    known_indices = get_known_indices(X)
    for trial in range(trials):
        _test_error_data = delayed(run_trial)(solver, X, known_indices, trial)
        errors += _test_error_data
    errors = compute(errors)
    # TODO dlaczego jest tuple?
    df = pd.DataFrame(data=errors[0], columns=['param', 'trial', 'shrinkage value', 'test error'])
    df.to_csv(os.path.join(OUTPUT_DIR, type(solver).__name__))
    return df
