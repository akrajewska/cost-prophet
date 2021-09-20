import numpy as np
import pandas as pd
import os

from cost_prophet.utils.evaluation import test_error
from cost_prophet.utils.linear_alebra import split_tests_sets, get_known_indices
from dask import delayed, compute
from time import time
from dotenv import dotenv_values
from collections import OrderedDict
from copy import deepcopy
from matrix_completion import svt_solve

config = dotenv_values()
OUTPUT_DIR = config.get("OUTPUT_DIR")

class ImputeRunner:

    def __init__(self, solver_cls, solver_kwargs, params):
        self.solver_cls = solver_cls
        self.solver_kwargs = solver_kwargs
        self.params = params

    def solve(self, X:np.ndarray, X_train: np.ndarray, param_set: OrderedDict, test_indices: list, trial:int) -> float:
        solver_kwargs = deepcopy(self.solver_kwargs) | param_set
        solver = self.solver_cls(**solver_kwargs)
        outputs = solver.fit_transform(X_train)
        results = self.tranform(outputs, X, param_set, test_indices, trial)
        return results

    def tranform(self, outputs: list, X:np.ndarray, param_set: OrderedDict, test_indices:list, trial:int):
        results = []
        for shrinkage_value, X_out in outputs:
            _test_error = test_error(X_out, X, test_indices)
            results.append( list(param_set.values()) +[trial, shrinkage_value, _test_error])
        _test_error = test_error(X_out, X, test_indices)
        return results

    def run_trial(self, X: np.ndarray, known_indices: list, param_set: OrderedDict, trial: int) -> list:
        test_indices, train_indices, X_train = split_tests_sets(known_indices, X)
        _test_error_data = self.solve(X, X_train, param_set, test_indices, trial)
        return _test_error_data

    def run(self, X: np.ndarray, trials: int):
        errors = []
        known_indices = get_known_indices(X)
        for param_set in self.params:
            for trial in range(trials):
                _test_error_data = delayed(self.run_trial)(X, known_indices, param_set, trial)
                errors+=_test_error_data
        errors = compute(errors)
        self.save_results(errors)

    def save_results(self, errors):
        columns = list(self.params[0].keys())
        columns += ['trial', 'shrinkage_value', 'test_error']
        df = pd.DataFrame(data=errors[0], columns=columns)
        df.to_csv(os.path.join(OUTPUT_DIR, f'{self.solver_cls.__name__}-{time()}'))


class SoftImputeRunner(ImputeRunner):

    def run_trial(self):
        pass

class SVTImputeRunner(ImputeRunner):

    def solve(self, X:np.ndarray, X_train: np.ndarray, param_set: OrderedDict, test_indices: list) -> float:
        mask_train = np.logical_not(np.isnan(X_train))
        solver_kwargs = deepcopy(self.solver_kwargs) | param_set
        X_out = self.solver_cls(X_train, mask_train, **solver_kwargs)
        _test_error = test_error(X_out, X, test_indices)
        return _test_error


    def run_trial(self, X: np.ndarray, known_indices: list, param_set: OrderedDict, trial: int) -> list:
        test_indices, train_indices, X_train = split_tests_sets(known_indices, X)
        _test_error = self.solve(X, X_train, param_set, test_indices)
        result = list(param_set.values()) + [trial, _test_error]
        return result



