import numpy as np

def get_ranks_grid(X:np.array, n:int) -> np.ndarray:
    max_rank = min(X.shape)
    return np.linspace(1, max_rank, n, dtype=int, endpoint=False)


def get_known_indices(X: np.ndarray) -> list:
    return np.argwhere(~np.isnan(X)).tolist()


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