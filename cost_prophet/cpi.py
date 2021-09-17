import os
from dask.distributed import Client
import pandas as pd

from fancyimpute import SoftImputeWarmStarts, SoftImpute
from cost_prophet.utils.experiment import experiment

DATA_DIR = '/home/tosia/cost_prophet/DATA'
OUTPUT_DIR = '/home/tosia/cost_prophet/OUTPUT'
trace_file_name = 'results-20210621-222050.json'
trace_file = os.path.join(DATA_DIR, trace_file_name)

df = pd.read_json(trace_file)
df = pd.pivot_table(df, values="f0_", index="machine_id", columns="collection_id")
X = df.to_numpy()
trials = 2
solver = SoftImputeWarmStarts(max_iters=1000, min_value=0.2, convergence_threshold=0.0001, init_fill_method='mean', verbose=False)

df = experiment(solver, trials, X)
df.to_csv(os.path.join(OUTPUT_DIR, 'softimpute.csv'))