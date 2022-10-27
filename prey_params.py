import setup
import numpy as np
import random

default_params_prey = {
    "position": (random.randrange(setup.GRID_WIDTH), random.randrange(setup.GRID_HEIGHT)),
    "food_target": None,
    "zl": 25,  # alignment zone
    "dr": 0.9,  # individual reach
    "max_speed": 0.1,
    "max_neighbour_awareness": 50,  # meters
    "h": 5,  # half-max distance detect predator
    "N": 5,  # scaling for predator detection
    "em": 1,  # metabolism
    "max_energy": 100000,  # called eM in paper
    "death_rate": 0.1,
    "max_age": 10512000,  # 60 * 24 * 365 * 20: 20 years expressed in minutes
    "mutation_rate": 0.05,
    "is_safe": True,
    "waiting_time": 10,  # TODO find initial value
    "reaction_time": 1,
    "er": 2,  # energy gained per food item TODO should this be in model?
    "t_min": 10,
    "te": 10,  # handling time
    "nrz": 0,  # number of actual neighbours
    "di": 0,  # the current direction/facing, degrees
    "v_hat": [0, 1],  # unit direction vector
}

evolvable_params_prey = {
    # descision making
    # predator scan, between 0 and 1, sd : 0.2
    "pv": np.random.normal(0.5, 0.2, 1),
    # move after move, between 0 and 1, sd : 0.2
    "pm": np.random.normal(0.5, 0.2, 1),
    # food scan after eat, between 0 and 1, sd : 0.2
    "pse": np.random.normal(0.5, 0.2, 1),
    # food scan after no food, between 0 and 1, sd : 0.2
    "psn": np.random.normal(0.5, 0.2, 1),
    # move to food, between 0 and 1, sd : 0.2
    "pmtf": np.random.normal(0.5, 0.2, 1),
    # vigilance
    # scan duration, between 0.167 and 1.99, sd : 0.4
    "tv": np.random.uniform(0.167, 1.99, 1),
    # scan angle, between 0 and 360, sd : 72
    "av": np.random.normal(180, 72, 1),
    # fleeing
    "tp": np.random.normal(10, 5, 1),  # flee duration, minimum 0, sd : 5
    # grouping
    # repulsion zone, between 0 and 50, sd : 10
    "zr": np.random.normal(25, 10, 1),
    # attractrion zone, between zr and 50, sd : 10
    "za": np.random.normal(40, 10, 1),
    # maximum turning angle for attraction, between 0 and 360, sd : 72
    "aa": np.random.normal(180, 72, 1),
    # maximum turning angle for repulsion, between 0 and 360, sd : 72
    "ar": np.random.normal(180, 72, 1),
    "nr": np.random.normal(5, 1, 1),  # tolerated neighbors, 0 min std 1
    # movement
    # move duration, between 0.167 and 1.99, sd : 0.4
    "tm": np.random.uniform(0.167, 1.99, 1),
    "dm":  np.random.normal(10, 3, 1),  # move distance, minimum 0, sd = 3
    # move angle, between 0 and 360, sd = 72
    "am": np.random.normal(180, 72, 1),
    # foraging
    "df": 2,  # search radius of forager
    "af": 270,  # search angle, angle between food and forward direction
    "tf": 3  # foodscan duration
}


def get_default_params_prey():
    return default_params_prey.copy()


def get_evolvable_params_prey():
    return evolvable_params_prey.copy()
