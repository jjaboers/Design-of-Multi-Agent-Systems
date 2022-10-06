from mesa.datacollection import DataCollector as DC

# TODO add more data 
# 	- total energy (per agent type)
# 	- number of deaths
# 	- number of kills
# 	- number of plants eaten 
# 	- etc

model_reporters = {
	"n_agents"		: 	lambda m: m.schedule.get_agent_count()	,
	"n_prey"		: 	lambda m: m.n_agents_per_type["prey"]	,
	"n_food"		: 	lambda m: m.n_agents_per_type["food"]	,
	"n_predator"	: 	lambda m: m.n_agents_per_type["predator"]	
}
agent_reporters = {
	"params"		:	"params"								,
}


class DataCollector(DC):
	def __init__(self, model):
		super().__init__(model_reporters = model_reporters, 
			agent_reporters = agent_reporters)
