import setup
import prey_params
import predator_params

model_params_5 = {
    "N": setup.N_AGENTS,
    "width": setup.GRID_WIDTH,
    "height": setup.GRID_HEIGHT,
    "dict_pred": predator_params.default_params_predator_5,
    "dict_prey_evolvable": prey_params.default_params_prey,
    "dict_prey_nonevolvable": prey_params.evolvable_params_prey
}

model_params_7 = {
    "N": setup.N_AGENTS,
    "width": setup.GRID_WIDTH,
    "height": setup.GRID_HEIGHT,
    "dict_pred": predator_params.default_params_predator_7,
    "dict_prey_evolvable": prey_params.default_params_prey,
    "dict_prey_nonevolvable": prey_params.evolvable_params_prey
}

model_params_9 = {
    "N": setup.N_AGENTS,
    "width": setup.GRID_WIDTH,
    "height": setup.GRID_HEIGHT,
    "dict_pred": predator_params.default_params_predator_9,
    "dict_prey_evolvable": prey_params.default_params_prey,
    "dict_prey_nonevolvable": prey_params.evolvable_params_prey
}
