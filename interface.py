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
        "r": 0.5,
    }
    if agent.type == "prey":
        portrayal["Color"] = "grey"
        print("prey")
        portrayal["Layer"] = 1
    elif agent.type == "food":
        portrayal["Color"] = "green"
        print("food")
        portrayal["Layer"] = 0
    elif agent.type == "predator":
        portrayal["Color"] = "red"
        print("predator")
        portrayal["Layer"] = 2

    return portrayal


grid = mesa.visualization.CanvasGrid(agent_portrayal, setup.UI_WIDTH, setup.UI_HEIGHT, 500, 500)
server = mesa.visualization.ModularServer(
    Model, [grid], "Prey-Predator Model", {"N": setup.N_AGENTS, "width": setup.UI_WIDTH, "height": setup.UI_HEIGHT}
)
server.port = 8521  # The default
server.launch()