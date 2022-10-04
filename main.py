import mesa
import matplotlib.pyplot as plt
import numpy as np
import random
import setup

from model import Model

model = Model(setup.N_AGENTS, setup.UI_WIDTH, setup.UI_HEIGHT)
for i in range(20):
    model.step()


agent_counts = np.zeros((model.grid.width, model.grid.height))
for cell in model.grid.coord_iter():
    cell_content, x, y = cell
    agent_count = len(cell_content)
    agent_counts[x][y] = agent_count
plt.imshow(agent_counts, interpolation="nearest")
plt.colorbar()
plt.show()
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
