from data_collector import DataCollector as DC
from matplotlib import pyplot as plt
import numpy as np

def plot_populations(dc):

	n_agents = dc.get_model_vars_dataframe()["n_agents"]
	n_prey = dc.get_model_vars_dataframe()["n_prey"]
	n_predators = dc.get_model_vars_dataframe()["n_predator"]
	
	x = np.arange(dc.get_model_vars_dataframe().shape[0])
	plt.plot(x, n_prey, n_predators)
	plt.title("populations over time")
	plt.legend(["n_prey", "n_predators"])
	plt.xlabel("time (steps)")
	plt.ylabel("population")
	plt.show()
	
def plot_food_count(dc):
	n_food = dc.get_model_vars_dataframe()["n_food"]
	x = np.arange(dc.get_model_vars_dataframe().shape[0])
	plt.plot(x, n_food)
	plt.title("food over time")
	plt.xlabel("time (steps)")
	plt.ylabel("n food")
	plt.show()