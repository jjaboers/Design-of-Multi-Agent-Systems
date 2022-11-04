import mesa
import matplotlib.pyplot as plt
import numpy as np
import random
import setup
import pandas as pd

from model import Model
from model_params import model_params_no_evolve_5, model_params_no_evolve_7, model_params_no_evolve_9, model_params_evolve_5, model_params_evolve_7, model_params_evolve_9
from data_collector import model_reporters
from plot_results import plot_populations
from mesa.batchrunner import FixedBatchRunner

batch_params = [model_params_no_evolve_5,
                model_params_no_evolve_7, model_params_evolve_9]

batch_runner = FixedBatchRunner(
    Model, batch_params, max_steps=20, model_reporters=model_reporters)

result_batch_run = batch_runner.run_all()

print(result_batch_run)

results_df = pd.DataFrame(result_batch_run)
print(results_df.keys())

