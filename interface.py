import mesa
from model import Model
import setup
from SimpleContinuousModule import SimpleCanvas


def agent_portrayal(agent):
    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "r": 1,
    }
    if agent.type == "prey":
        portrayal["Color"] = "blue"
        portrayal["Layer"] = 1
        portrayal["r"] = 4
    elif agent.type == "food":
        portrayal["Color"] = "green"
        portrayal["Layer"] = 0
    elif agent.type == "predator":
        portrayal["Color"] = "red"
        portrayal["Layer"] = 2
        portrayal["r"] = 4

    return portrayal


grid = SimpleCanvas(agent_portrayal, 500, 500)


server = mesa.visualization.ModularServer(
    Model, [grid], "Prey-Predator Model", {"N": setup.N_AGENTS, "width": setup.GRID_WIDTH, "height": setup.GRID_HEIGHT, "attack_distance": 5, "evolve": True}
)


server.port = 8521  # The default
server.launch()