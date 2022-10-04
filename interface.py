"""
The full code should now look like:
"""
# from MoneyModel import *
import mesa
from model import Model
import setup

def agent_portrayal(agent):
    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "Layer": 0,
        "Color": "red",
        "r": 0.5,
    }
    return portrayal


grid = mesa.visualization.CanvasGrid(agent_portrayal, setup.UI_WIDTH, setup.UI_HEIGHT, 500, 500)
server = mesa.visualization.ModularServer(
    Model, [grid], "Prey-Predator Model", {"N": setup.N_AGENTS, "width": setup.UI_WIDTH, "height": setup.UI_HEIGHT}
)
server.port = 8521  # The default
server.launch()