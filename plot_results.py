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


def plot_A(dfs):
	x = []
	y = []
	for df in dfs:
		x.append(df["predation_risk"].unique())
		y.append(np.mean(df["vigilance_avg"]))
	
	plt.plot(x, y)
	plt.title("vigilance vs predation risk")
	plt.xlabel("predation risk")
	plt.ylabel("vigilance")
	plt.show()

def plot_B(dfs):
	x = []
	y = []
	for df in dfs:
		x.append(df["predation_risk"].unique())
		y.append(np.mean(df["group_size_prey"]))
	
	plt.plot(x, y)
	plt.title("group size vs predation risk")
	plt.xlabel("predation risk")
	plt.ylabel("group size prey")
	plt.show()

def plot_C(df):
	x = df["time"]
	y = df["vigilange_avg"]

	plt.plot(x, y)
	plt.title("vigilance over time d_p = " + str(df["predation_risk"]))
	plt.xlabel("time (mins)")
	plt.ylabel("vigilance")
	plt.show()

def plot_D(df):
	x = df["time"]
	y = df["group_size_prey"]

	plt.plot(x, y)
	plt.title("vigilance over time d_p = " + str(df["predation_risk"]))
	plt.xlabel("time (mins)")
	plt.ylabel("group size prey")
	plt.show()