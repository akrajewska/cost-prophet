import os
from collections import OrderedDict

import pandas as pd
from dotenv import dotenv_values
from fancyimpute import IterativeSVD

from cost_prophet.imput_runner import ImputeRunner
# TODO dlaczego nie dzialaja relative imports
from cost_prophet.utils.linear_alebra import get_ranks_grid

config = dotenv_values()

DATA_DIR = config.get("DATA_DIR")

trace_file_name = 'results-20210621-222050.json'
trace_file = os.path.join(DATA_DIR, trace_file_name)

df = pd.read_json(trace_file)
df = pd.pivot_table(df, values="f0_", index="machine_id", columns="collection_id")
X = df.to_numpy()
trials = 2
# solver = SoftImputeWarmStarts(max_iters=1, min_value=0.2, convergence_threshold=0.0001, init_fill_method='mean', verbose=False)
# df = experiment(solver, trials, X)

# solver_kwargs = OrderedDict({"max_iters":100,
#           "min_value": 0.2,
#           "convergence_threshold": 0.0001,
#           "init_fill_method": "mean",
#           "verbose": False})
#
#
# runner = ImputeRunner(solver_cls=SoftImputeWarmStarts, solver_kwargs=solver_kwargs, params=[{'shrinkage_values_number': 10}])
# runner.run(X, 5)


ranks = get_ranks_grid(X, 50)
solver_kwargs = OrderedDict({"max_iters": 100,
                             "min_value": 0.2,
                             "convergence_threshold": 0.0001,
                             "init_fill_method": "mean",
                             "verbose": False,
                             "gradual_rank_increase": False})
params = [{'rank': rank} for rank in ranks]

runner = ImputeRunner(solver_cls=IterativeSVD, solver_kwargs=solver_kwargs, params=params)
runner.run(X, 5)

# runner = SVTImputeRunner(svt_solve, {'kick_device': True}, [{'tau': None}])
# runner.run(X, 2)
