from model_params import *
import pandas as pd
from model import Model
import time
import numpy as np

batch_params = [model_params_no_evolve_5,
                model_params_no_evolve_7, 
				model_params_no_evolve_9,
				model_params_evolve_5,
                model_params_evolve_7, 
				model_params_evolve_9]
class BatchRun:
	def __init__(self, n_timesteps, runs_per_setup = 3, batch_params=batch_params) -> None:
		self.time_step = 0
		self.n_time_steps = n_timesteps
		self.batch_params = batch_params
		self.runs_per_setup = runs_per_setup
		self.batch_results = []
		

	def run(self, save = True):
		for batch_idx in range(len(self.batch_params)):
			print("Batch " + str(batch_idx))
			t_start = time.time()
			N, width, height, attack_distance, evolve = self.batch_params[batch_idx].values()
			data_this_setup = []
			for run in range(self.runs_per_setup):
				print("run " + str(run))
				model = Model(N, width, height, 
									attack_distance, evolve)
				for t_step in range(self.n_time_steps):
					model.step()
				
				data_this_setup.append(model.get_global_overview())
			df = self.average_entries(data_this_setup)
			print(df.shape)
			
			self.batch_results.append(df)
			if save:
				df.to_csv("./results/batch{0}_{1}.csv".format(batch_idx, self.n_time_steps))
				
			print("batch " + str(batch_idx) + " completed in: " + str(time.time() - t_start) + " s")
		# if save:
		# 	batch_idx = 0
		# 	for df_ in self.batch_results:
		# 		df_.to_csv("./results/batch{0}.csv".format(batch_idx))
		# 		batch_idx += 1


	def average_entries(self, dfs):
		final_df = pd.DataFrame(columns = dfs[0].columns)
		
		for row in range(dfs[0].shape[0]):
			predation_risk = 0.0
			vigilance_total = 0.0
			vigilance_avg = 0.0
			time = 0
			group_size_prey = 0.0
			for df in dfs:
				predation_risk += df["predation_risk"].to_numpy()[row] # TODO might be stupid to average out
				vigilance_total += df["vigilance_total"].to_numpy()[row]
				vigilance_avg += df["vigilance_avg"].to_numpy()[row]
				time += df["time"].to_numpy()[row] # kind of stupid I know
				group_size_prey += df["group_size_prey"].to_numpy()[row]
			predation_risk /= len(dfs)
			vigilance_total /= len(dfs)
			vigilance_avg /= len(dfs)
			time /= len(dfs)
			group_size_prey /= len(dfs)
			data = {	"predation_risk" : predation_risk,
					"vigilance_total" : vigilance_total,
					"vigilance_avg" : vigilance_avg, 
					"time"			: time,
					"group_size_prey" : group_size_prey
				}
			row_add = pd.DataFrame(data, index=[0])
			final_df = pd.concat([final_df, row_add])
			# print(final_df)
		return final_df


br = BatchRun(n_timesteps=2000, runs_per_setup = 3, batch_params=batch_params)
br.run(save=True)