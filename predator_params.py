import setup
import numpy.random as random
from copy import deepcopy

SCALED_FLAG = False

# this way all params can be manipulated in higher levels
# Check what is evolvable
default_params_predator = {
    "position":   (0, 0),
    "initial_energy":   100000,
    "search_angle":   250,   # degrees TODO implement
    "t_food_scan":   (0.167 + 1.99) / 2,
    "alignment":   50, 
    "reach":   1.0,
    "max_speed":   2,   # TODO find right val
    "max_neighbour_awareness":   50, # TODO add to evolvable
    "energy_cost":   1,
    "max_energy":   100000,
    "death_rate":   0.1,
    "max_age":   10512000,   # 60*24*365*20 = 20years in mins
    "mutation_rate":   0.05,
    "reproduction_requirement":   100000,   # max energy
    "reproduction_cost":   50000,   # half of max energy (paper)
    "offspring_energy":   50000,   # half of max energy (paper)
    "r_repulsion":   25,
    "r_attraction":   25,
    "angle_repulsion":   180,
    "angle_attraction":   180,
    "prey_detection_range":   50,   # not sure because angle and r
    "attack_speed":   11.1,   # (m/s) prey paper, check wolf paper
    "search_duration":   3,
    "angle_move"    :   180 
}
# Also works with initialization if you pass default params
def mutate_params(params):
    params = deepcopy(params)
    if random.random() < 0.05:
        params["t_food_scan"] = random.uniform(0.167, 1.99, 1)
    if random.random() < 0.05:
        params["r_repulsion"] = random.normal(params["r_repulsion"], 10, 1)
    if random.random() < 0.05:
        params["r_attraction"] = random.normal(params["r_attraction"], 10, 1)
        params["r_attraction"] = min(params["r_attraction"], params["r_repulsion"])
    if random.random() < 0.05:
        params["t_food_scan"] = random.uniform(0.167, 1.99, 1)
    if random.random() < 0.05:
        params["angle_repulsion"] = random.normal(params["angle_repulsion"], 72, 1)
    if random.random() < 0.05:
        params["angle_attraction"] = random.normal(params["angle_attraction"], 72, 1)
    if random.random() < 0.05:
        params["angle_move"] = random.normal(params["angle_move"], 72, 1)
    
    
    return params


def get_default_params_predator():
    return deepcopy(default_params_predator)

