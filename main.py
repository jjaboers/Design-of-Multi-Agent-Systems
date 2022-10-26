import mesa
import matplotlib.pyplot as plt
import numpy as np
import random
import setup
import pandas as pd

from model import Model
from model_params import model_params
from plot_results import plot_populations
from mesa.batchrunner import FixedBatchRunner

batch_params = [model_params]

batch_runner = FixedBatchRunner(Model, batch_params)

result_batch_run = batch_runner.run_all()

print(result_batch_run)

results_df = pd.DataFrame(result_batch_run)
print(results_df.keys())

# our_model = Model(model_params)


# # for i in range(20000000):
# #     model.step()
# #params = {"N": setup.N_AGENTS, "width": setup.GRID_WIDTH, "height": setup.GRID_HEIGHT}
# result_batch_run = mesa.batch_run(Model, model_params, max_steps=10)



# plot_populations(model.datacollector)  

# print(model.datacollector.get_model_vars_dataframe())
# print(model.datacollector.get_agent_vars_dataframe())
# print(model.datacollector.evolvable_params_predator)
# print(model.datacollector.evolvable_params_prey)py

# agent_counts = np.zeros((model.grid.width, model.grid.height))
# for cell in model.grid.coord_iter():
#     cell_content, x, y = cell
#     agent_count = len(cell_content)
#     agent_counts[x][y] = agent_count
# plt.imshow(agent_counts, interpolation="nearest")
# plt.colorbar()
# plt.show()



# NOTE from tutorial:
# if __name__ == '__main__':

#     agents_per_process = 3
#     c = 0
#     agents = list()
#     for i in range(agents_per_process):
#         port = int(argv[1]) + c
#         agent_name = 'agente_hello_{}@localhost:{}'.format(port, port)
#         agente_hello = AgenteHelloWorld(AID(name=agent_name))
#         agents.append(agente_hello)
#         c += 1000

#     start_loop(agents)
