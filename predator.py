import mesa
from enum import Enum
import numpy as np
from TypedAgent import TypedAgent
from prey import Prey_State
import predator_params
from scipy.spatial.distance import euclidean as dist
import setup
from scipy.stats import truncnorm
from copy import deepcopy
import math
import random
# searching is roaming while scanning is looking

def trunc_normal(lower, upper, sd, mu):
    r = truncnorm.rvs(
        (lower - mu) / sd, (upper - mu) / sd, loc=mu, scale=sd, size=1)
    return r


def mutate(params):
    # print("params are " , params)
    for parameter in params:
        if random.random() < 0.05:
            if "zr" in parameter:
                params[parameter] = trunc_normal(0, 50, 10, params[parameter])
            elif "za" in parameter:
                params[parameter] = trunc_normal(
                    params["zr"], 50, 10, params[parameter])
            elif "a" in parameter:
                params[parameter] = trunc_normal(0, 360, 72, params[parameter])
            elif "tp" in parameter:
                pass
            elif "tv" in parameter:
                params[parameter] = trunc_normal(
                    0.167, 1.99, 0.4, params[parameter])
            elif "tm" in parameter:
                params[parameter] = trunc_normal(
                    0.167, 1.99, 0.4, params[parameter])
            elif "p" in parameter:
                params[parameter] = trunc_normal(0, 1, 0.2, params[parameter])
            # elif "n" in parameter:
            #     pass
            elif "dm" in parameter:
                params[parameter] = trunc_normal(0, 100, 3, params[parameter])
            elif "pv" in parameter:
                params[parameter] = trunc_normal(0, 1, 0.2, params[parameter])
            elif "pm" in parameter:
                params[parameter] = trunc_normal(0, 1, 0.2, params[parameter])
            elif "pse" in parameter:
                params[parameter] = trunc_normal(0, 1, 0.2, params[parameter])
            elif "psn" in parameter:
                params[parameter] = trunc_normal(0, 1, 0.2, params[parameter])
            elif "pmtf" in parameter:
                params[parameter] = trunc_normal(0, 1, 0.2, params[parameter])
            elif "ar" in parameter:
                params[parameter] = trunc_normal(0, 360, 72, params[parameter])
            elif "aa" in parameter:
                params[parameter] = trunc_normal(0, 360, 72, params[parameter])
            elif "nr" in parameter:
                params[parameter] = trunc_normal(0, 100, 1, params[parameter])
        # return at the end modified params
        return params

class Predator_State(Enum):
    SEARCHING = 1
    CHASING = 2
    SCANNING = 3
    EATING = 4
    DEAD = 5


class PredatorAgent(TypedAgent):
    """An agent that is a predator"""

    def __init__(self, unique_id, model, attack_distance,
                 params=predator_params.default_params_predator, evolve=False):
        super().__init__(unique_id, model, params)
        # non-evolvable parameters

        # not variable parameters, these are always the same at construction
        self.type = "predator"
        self.state = Predator_State.SEARCHING
        self.min_energy = 0
        self.t_current_activity = 0
        # will be a prey object
        self.target = None
        # for moving to a location will be a tuple
        self.destination = None
        self.model = model
        self.nearby_predators = []
        self.nearby_prey = []
        self.evolve = evolve

        # constants-------------------------------------------------------------
        # called eM in paper
        self.max_energy = params["max_energy"]
        self.death_rate = params["death_rate"]
        self.max_age = params["max_age"]
        self.mutation_rate = params["mutation_rate"]
        self.reproduction_requirement = params["reproduction_requirement"]
        self.reproduction_cost = params["reproduction_cost"]
        self.offspring_energy = params["offspring_energy"]
        # constants-------------------------------------------------------------

        # internal state--------------------------------------------------------
        self.position = params["position"]
        self.energy = params["initial_energy"]
        # internal state--------------------------------------------------------

        # perception------------------------------------------------------------
        # search angle between food and forward direction see sources
        self.search_angle = params["search_angle"]
        # foodscan duration
        self.t_food_scan = params["t_food_scan"]
        # meters
        self.max_neighbour_awareness = params["max_neighbour_awareness"]
        # perception------------------------------------------------------------

        # moving----------------------------------------------------------------
        # alignment zone
        self.alignment = params["alignment"]
        # individual reach
        self.reach = params["reach"]
        # higher than prey (see wolf paper)
        self.max_speed = params["max_speed"]
        # metabolism
        self.energy_cost = params["energy_cost"]
        # roaming time
        self.search_duration = params["search_duration"]
        # moving----------------------------------------------------------------

        # evolvable parameters--------------------------------------------------
        self.r_repulsion = params["r_repulsion"]
        self.r_attraction = params["r_attraction"]
        self.angle_repulsion = params["angle_repulsion"]
        self.angle_attraction = params["angle_attraction"]
        # evolvable parameters--------------------------------------------------

        # predator specific parameters------------------------------------------
        self.attack_distance = attack_distance
        self.prey_detection_range = params["prey_detection_range"]
        self.attack_speed = params["attack_speed"]
        # predator specific parameters------------------------------------------

    # TODO check duration of step as age is in minutes
    def step(self):
        if self.evolve:
            if self.age >= self.max_age:
                self.die()
            # TODO implement death_rate
            self.reproduce()
        if self.state == Predator_State.DEAD:
            return
        elif self.state == Predator_State.SEARCHING:
            self.search()
        elif self.state == Predator_State.CHASING:
            self.chase()
        elif self.state == Predator_State.SCANNING:
            self.scan()
        elif self.state == Predator_State.EATING:
            self.eat()

        self.t_current_activity += 1
        self.age += 1
        self.energy -= self.energy_cost

    # random movement with scanning inbetween

    def search(self):
        if self.t_current_activity >= self.search_duration:
            self.set_state(Predator_State.SCANNING)
            return

        # possible_steps =( (self.position[0] + np.random.random() * self.max_speed).round(decimals=2),
        #                   (self.position[1] + np.random.random() * self.max_speed).round(decimals=2) )
        # print(possible_steps)
            # self.model.grid.get_neighborhood(
            # self.position,
            # moore=True,
            # include_center=False)
        # new_position = self.random.choice(possible_steps)
        new_position = (round((self.position[0] + np.random.random() * self.max_speed), 2),
                        round((self.position[1] + np.random.random() * self.max_speed), 2))
        # print(new_position)
        self.move(new_position)

    # with current fleeing system more like charge
    def chase(self):
        if self.target == None or not self.target.is_alive():
            self.set_state(Predator_State.SEARCHING)
            return
        if self.target.is_safe:
            self.set_state(Predator_State.SEARCHING)
            return
        if dist(self.position, self.target.get_position()) <= self.attack_distance:
            self.set_state(Predator_State.EATING)
            return

        for step in range(self.max_speed):
            possible_steps = self.model.grid.get_neighborhood(
                self.position,
                moore=True,
                include_center=True)
            # Select position closest to target position
            new_position = possible_steps[
                np.argmin(
                    [
                        dist(self.target.get_position(), pos)
                        for pos in possible_steps
                    ]
                )
            ]
            self.move(new_position)

    def scan(self):
        if self.t_current_activity >= self.search_duration:
            self.set_state(Predator_State.SEARCHING)
            return
        agent = self.model.get_closest_agent_of_type_in_range(
            pos=self.position,
            type="prey",
            range=self.prey_detection_range
        )
        if agent != None:
            self.target = agent
            self.set_state(Predator_State.CHASING)

    def eat(self):
        self.energy += self.target.get_energy()
        print("target is ", self.target)
        print(len(self.target))
        if self.energy < self.max_energy:
            self.energy = self.max_energy
        self.target.set_state(Prey_State.DEAD)
        self.model.schedule.remove(self.target)
        self.set_state(Predator_State.SEARCHING)

    # TODO implement repulsion etc

    def move(self, new_position):
        print("new_position: ", new_position)
        self.model.grid.move_agent(self, new_position)
        self.set_position(new_position)

    def set_target(self, target):
        self.target = target

    def set_state(self, state):
        self.state = state
        self.t_current_activity = 0

    def get_state(self):
        return self.state

    def is_alive(self):
        return self.state != Predator_State.DEAD

    # asexual reproduction

    def reproduce(self):
        if self.energy < self.reproduction_requirement:
            return
        self.energy -= self.reproduction_cost
        params = predator_params.mutate_params(deepcopy(self.params))
        self.model.create_new_predator(params)
        # if self.evolve:
        #     params = predator_params.mutate_params(self.params)
        #     self.model.create_new_predator(params)
        # else:
        #     self.model.create_new_predator(self.params)

    def die(self):
        super().die()
        self.set_state(Predator_State.DEAD)

    # def reproduce(self):
    #     # Reproduction
    #     self.energy = self.energy - self.max_energy / 2
    #     #print("self energy step1 ", self.energy)
    #     # TODO params is the baseclass name maybe use evolvable
    #     child_params = mutate(deepcopy(self.evolvable_params))
    #     #print("child params ", child_params)
    #     self.model.create_new_prey(child_params)
    #     #print("created new prey!!!")

    def force_birth(self):
        n = 5
        # print("self energy step force birth ", self.energy)
        summed_energy_neighbours = 0
        for agent in self.model.grid.get_neighbors((int(setup.GRID_WIDTH/2), int(setup.GRID_HEIGHT/2)),
                                                   include_center=True,
                                                   radius=setup.GRID_WIDTH+1):
            if agent.type == "prey":
                # print("agent energy ", agent.energy)
                summed_energy_neighbours += agent.energy
        summed_energy_neighbours = max(summed_energy_neighbours, 1)
        prob_to_birth = math.pow(self.energy / summed_energy_neighbours, n)
        # print(" prob to birth ", str(prob_to_birth))
        if np.random.random() > prob_to_birth:
            self.reproduce()
