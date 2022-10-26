import setup 
import prey_params
import predator_params

model_params = {
    "N": setup.N_AGENTS, 
    "width": setup.GRID_WIDTH, 
    "height": setup.GRID_HEIGHT,
    "dict_pred" : predator_params.default_params_predator,
    "dict_prey_evolvable": prey_params.default_params_prey,
    "dict_prey_nonevolvable": prey_params.evolvable_params_prey
}