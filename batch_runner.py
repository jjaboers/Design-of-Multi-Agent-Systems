from model_params import *
import pandas as pd
from model import Model

batch_params = [model_params_no_evolve_5,
                model_params_no_evolve_7, 
				model_params_no_evolve_9,
				model_params_evolve_5,
                model_params_evolve_7, 
				model_params_evolve_9]
class BatchRun:
	def __init__(self, n_timesteps, runs_per_setup = 3, batch_params=batch_params) -> None:
		self.time_step = 0
		self.n_time_step = n_timesteps
		self.batch_params = batch_params
		self.runs_per_setup = runs_per_setup
		self.batch_results = []
		

	def run(self, save = True):
		for batch_idx in len(self.batch_params):
			N, width, height, attack_distance, evolve = self.batch_params[batch_idx]
			data_this_setup = []
			for run in range(self.runs_per_setup):
				model = Model(N, width, height, 
									attack_distance, evolve)
				for t_step in range(self.n_time_steps):
					model.step()
				data_this_setup.append(model.get_global_overview())
			df = self.average_entries(data_this_setup)
			self.batch_results.append(df)
		if save:
			batch_idx = 0
			for df_ in self.batch_results:
				df_.to_csv("./results/batch{0}.csv".format(batch_idx))
				batch_idx += 1


	def average_entries(self, dfs):
		final_df = pd.DataFrame(columns = dfs[0].columns)
		for row in len(dfs[0]):
			predation_risk = 0.0
			vigilance_total = 0.0
			vigilance_avg = 0.0
			time = 0
			group_size_prey = 0.0
			for df in dfs:
				predation_risk += df["predation_risk"] # TODO might be stupid to average out
				vigilance_total += df["vigilance_total"]
				vigilance_avg += df["vigilance_avg"]
				time += df["time"] # kind of stupid I know
				group_size_prey += 0.0
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
			row_add = pd.DataFrame(data)
			final_df = pd.concat([final_df, row_add])
		return final_df


br = BatchRun(n_timesteps=20000, runs_per_setup = 3, batch_params=batch_params)
br.run(save=True)