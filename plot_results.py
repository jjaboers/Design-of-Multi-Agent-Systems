from data_collector import DataCollector as DC
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

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


def plot_A(dfs, path):
	x = []
	y = []
	for df in dfs:
		x.append(df["predation_risk"][0])
		y.append(np.mean(df["vigilance_avg"]))
	
	plt.bar(x, y)
	plt.title("vigilance vs predation risk")
	plt.xlabel("predation risk")
	plt.ylabel("vigilance")
	plt.savefig(path, format='png')
	plt.show()

def plot_B(dfs, path):
	x = []
	y = []
	for df in dfs:
		x.append(df["predation_risk"][0])
		y.append(np.mean(df["group_size_prey"]))
	
	plt.bar(x, y)
	plt.title("group size vs predation risk")
	plt.xlabel("predation risk")
	plt.ylabel("group size prey")
	plt.savefig(path, format='png')
	plt.show()

def plot_C(df, path):
	x = df["time"]
	y = df["vigilance_avg"]

	plt.plot(x, y)
	plt.title("vigilance over time d_p = " + str(df["predation_risk"].unique()))
	plt.xlabel("time (mins)")
	plt.ylabel("vigilance")
	plt.savefig(path, format='png')
	plt.show()

def plot_D(df, path):
	x = df["time"]
	# print(x)
	y = df["group_size_prey"]
	# print(y)

	plt.plot(x, y)
	plt.title("group size over time d_p = " + str(df["predation_risk"].unique()))
	plt.xlabel("time (mins)")
	plt.ylabel("group size prey")
	plt.savefig(path, format='png')
	plt.show()


dfs = []
for idx in range(6):
	df = pd.read_csv("./results/batch{0}_300.csv".format(idx))
	dfs.append(df)

plot_A(dfs[:3], "./results/vig_vs_pred_evolve.png")

# input("press button")

plot_A(dfs[3:], "./results/vig_vs_pred.png")

# input("press button")

plot_B(dfs[:3], "./results/group_sz_vs_pred_evolve.png")
# input("press button")

plot_B(dfs[3:], "./results/group_sz_vs_pred.png")
# input("press button")

plot_C(dfs[3], "./results/vig_over_time.png")
# input("press button")

plot_D(dfs[3], "./results/group_size_over_time.png")
# input("press button")

plot_C(dfs[0], "./results/vig_over_time_evolve.png")
input("press button")

plot_D(dfs[0], "./results/group_size_over_time_evolve.png")
# input("press button")