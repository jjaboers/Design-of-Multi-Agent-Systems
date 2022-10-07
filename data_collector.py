from mesa.datacollection import DataCollector as DC
import pandas as pd
import prey 
import predator


model_reporters = {
	"n_agents"		: 	lambda m: m.schedule.get_agent_count()	,
	"n_prey"		: 	lambda m: m.n_agents_per_type["prey"]	,
	"n_food"		: 	lambda m: m.n_agents_per_type["food"]	,
	"n_predator"	: 	lambda m: m.n_agents_per_type["predator"]	
}
agent_reporters = None


class DataCollector(DC):
	
	def __init__(self, model, model_reporters = model_reporters, 
			agent_reporters = agent_reporters):
		super().__init__(model_reporters = model_reporters, 
			agent_reporters = agent_reporters)
		self.evolvable_params_prey = pd.DataFrame()
		self.evolvable_params_predator = pd.DataFrame()
		#  Initial dataf
		self.evolvable_params_prey.insert(loc = 0, column="id", value=-1)
		self.evolvable_params_predator.insert(loc = 0, column="id", value=-1)


	def collect(self, model):
		super().collect(model)
		self.record_evolvable_params(model)
	
	def record_evolvable_params(self, model):
		agents = model.schedule.agent_buffer()
		for agent in agents:
			if agent.type == "prey":
				prey_dict = agent.evolvable_params.copy()
				prey_dict["unique_id"] = agent.unique_id
				if self.evolvable_params_prey.empty:
					self.evolvable_params_prey = pd.DataFrame([prey_dict])
					continue
				if agent.unique_id in self.evolvable_params_prey["unique_id"].values:
					continue
				df = pd.DataFrame([prey_dict])
				self.evolvable_params_prey = pd.concat([self.evolvable_params_prey, df])
			
			if agent.type == "predator":
				pred_dict = agent.params.copy()
				pred_dict["unique_id"] = agent.unique_id
				if self.evolvable_params_predator.empty:
					self.evolvable_params_predator = pd.DataFrame([pred_dict])
					continue
				if agent.unique_id in self.evolvable_params_predator["unique_id"].values:
					continue
				
				df = pd.DataFrame([pred_dict])
				self.evolvable_params_predator = pd.concat([self.evolvable_params_predator, df])
