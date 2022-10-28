import setup
import prey_params
import predator_params

model_params_evolve_5 = {
    "N": setup.N_AGENTS,
    "width": setup.GRID_WIDTH,
    "height": setup.GRID_HEIGHT,
    "attack_distance": 5,
    "evolve": True
}

model_params_evolve_7 = {
    "N": setup.N_AGENTS,
    "width": setup.GRID_WIDTH,
    "height": setup.GRID_HEIGHT,
    "attack_distance": 7,
    "evolve": True
}

model_params_evolve_9 = {
    "N": setup.N_AGENTS,
    "width": setup.GRID_WIDTH,
    "height": setup.GRID_HEIGHT,
    "attack_distance": 9,
    "evolve": True
}

model_params_no_evolve_5 = {
    "N": setup.N_AGENTS,
    "width": setup.GRID_WIDTH,
    "height": setup.GRID_HEIGHT,
    "attack_distance": 5,
    "evolve": False
}

model_params_no_evolve_7 = {
    "N": setup.N_AGENTS,
    "width": setup.GRID_WIDTH,
    "height": setup.GRID_HEIGHT,
    "attack_distance": 7,
    "evolve": False
}

model_params_no_evolve_9 = {
    "N": setup.N_AGENTS,
    "width": setup.GRID_WIDTH,
    "height": setup.GRID_HEIGHT,
    "attack_distance": 9,
    "evolve": False
}
