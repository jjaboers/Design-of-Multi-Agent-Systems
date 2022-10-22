import math
import mesa
from enum import Enum
from TypedAgent import TypedAgent

import numpy as np
import random
from scipy import spatial
import setup
from scipy.stats import truncnorm
from copy import deepcopy


class Prey_State(Enum):
    NOTHING = 1  # TODO called "NORMAL" in paper?
    MOVING = 3
    FOODSCAN = 4
    MOVETOFOOD = 5
    EATING = 6
    SCANNING = 7
    FLEEING = 8
    DEAD = 9


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
    "nrz": 0, # number of actual neighbours
    "di": 0, # the current direction/facing, degrees
    "v_hat": [0, 0], # unit direction vector
}

evolvable_params_prey = {
    # descision making
    "pv": np.random.normal(0.5, 0.2, 1),  # predator scan, between 0 and 1, sd : 0.2
    "pm": np.random.normal(0.5, 0.2, 1),  # move after move, between 0 and 1, sd : 0.2
    "pse": np.random.normal(0.5, 0.2, 1),  # food scan after eat, between 0 and 1, sd : 0.2
    "psn": np.random.normal(0.5, 0.2, 1),  # food scan after no food, between 0 and 1, sd : 0.2
    "pmtf": np.random.normal(0.5, 0.2, 1),  # move to food, between 0 and 1, sd : 0.2
    # vigilance
    "tv": np.random.uniform(0.167, 1.99, 1),  # scan duration, between 0.167 and 1.99, sd : 0.4
    "av": np.random.normal(180, 72, 1),  # scan angle, between 0 and 360, sd : 72
    # fleeing
    "tp": np.random.normal(10 ,5, 1),  # flee duration, minimum 0, sd : 5
    # grouping
    "zr": np.random.normal(25 , 10, 1),  # repulsion zone, between 0 and 50, sd : 10
    "za": np.random.normal(40 , 10, 1),  # attractrion zone, between zr and 50, sd : 10
    "aa": np.random.normal(180, 72, 1),  # maximum turning angle for attraction, between 0 and 360, sd : 72
    "ar": np.random.normal(180, 72, 1),  # maximum turning angle for repulsion, between 0 and 360, sd : 72
    "nr": np.random.normal(5, 1, 1),  # tolerated neighbors, 0 min std 1
    # movement
    "tm": np.random.uniform(0.167, 1.99, 1),  # move duration, between 0.167 and 1.99, sd : 0.4
    "dm":  np.random.normal(10, 3, 1),  # move distance, minimum 0, sd = 3
    "am": np.random.normal(180, 72, 1),  # move angle, between 0 and 360, sd = 72
    # foraging
    "df": 2,  # search radius of forager
    "af": 270,  # search angle, angle between food and forward direction
    "tf": 3  # foodscan duration
}


def get_default_params_prey():
    return default_params_prey.copy()


def get_evolvable_params_prey():
    return evolvable_params_prey.copy()


# Truncated normal distribution, takes range [lower, upper] and standard deviation (sd)
def trunc_normal(lower, upper, sd, mu):
    r = truncnorm.rvs(
        (lower - mu) / sd, (upper - mu) / sd, loc=mu, scale=sd, size=1)
    return r


def mutate(params):
    print("params are " , params)
    for parameter in params:
        if random.random() < 0.05:
            if "zr" in parameter:
                params[parameter] = trunc_normal(0, 50, 10, params[parameter])
            elif "za" in parameter:
                params[parameter] = trunc_normal(params["zr"], 50, 10, params[parameter])
            elif "a" in parameter:
                params[parameter] = trunc_normal(0, 360, 72, params[parameter])
            elif "tp" in parameter:
                pass
            elif "tv" in parameter:
                params[parameter] = trunc_normal(0.167, 1.99, 0.4, params[parameter])
            elif "tm" in parameter:
                params[parameter] = trunc_normal(0.167, 1.99, 0.4, params[parameter])
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



class PreyAgent(TypedAgent):
    """An agent that is a prey, as described in the paper."""

    def __init__(self, unique_id, model, default_params=default_params_prey, evolvable_params=evolvable_params_prey):
        super().__init__(unique_id, model)
        self.type = "prey"
        # self.model = model
        self.state = Prey_State.NOTHING
        self.previous_state = Prey_State.NOTHING
        self.current_action_time_remaining = 0
        self.detected_predator = False  # keep it like this or make it Boolean ?
        self.age = 100
        self.energy = 100000
        self.min_energy = 0
        self.default_params = default_params
        self.evolvable_params = evolvable_params

        self.position = default_params["position"]
        self.food_target = default_params["food_target"]
        self.zl = int(default_params["zl"])
        self.dr = int(default_params["dr"] * setup.PROPORTION)
        self.max_speed = int(default_params["max_speed"] * setup.PROPORTION)
        self.max_neighbour_awareness = int(default_params["max_neighbour_awareness"] * setup.PROPORTION)
        self.h = int(default_params["h"]) if int(default_params["h"]) > 5 else 5
        self.N = default_params["N"]
        self.em = default_params["em"]
        self.max_energy = default_params["max_energy"]
        self.death_rate = default_params["death_rate"]
        self.max_age = default_params["max_age"]
        self.mutation_rate = default_params["mutation_rate"]
        self.is_safe = default_params["is_safe"]
        self.waiting_time = default_params["waiting_time"]
        self.reaction_time = default_params["reaction_time"]
        self.er = default_params["er"]
        self.t_min = default_params["t_min"]
        self.te = default_params["te"]
        self.nrz = default_params["nrz"]
        self.di = default_params["di"]

        # descision making
        self.pv = evolvable_params["pv"]  # predator scan, between 0 and 1, sd = 0.2
        self.pm = evolvable_params["pm"]  # move after move, between 0 and 1, sd = 0.2
        self.pse = evolvable_params["pse"]  # food scan after eat, between 0 and 1, sd = 0.2
        self.psn = evolvable_params["psn"]  # food scan after no food, between 0 and 1, sd = 0.2
        self.pmtf = evolvable_params["pmtf"]  # move to food, between 0 and 1, sd = 0.2
        # vigilance
        self.tv = evolvable_params["tv"]  # scan duration, between 0.167 and 1.99, sd = 0.4
        self.av = evolvable_params["av"]  # scan angle, between 0 and 360, sd = 72
        # fleeing
        self.tp = evolvable_params["tp"]  # flee duration, minimum 0, sd = 5
        # grouping
        self.zr = evolvable_params["zr"]   # repulsion zone, between 0 and 50, sd = 10
        self.za = evolvable_params["za"]   # attractrion zone, between zr and 50, sd = 10
        self.aa = evolvable_params["aa"]  # maximum turning angle for attraction, between 0 and 360, sd = 72
        self.ar = evolvable_params["ar"]  # maximum turning angle for repulsion, between 0 and 360, sd = 72
        self.nr = evolvable_params["nr"]  # tolerated neighbors, (0 or 1)
        # movement
        self.tm = evolvable_params["tm"]  # move duration, between 0.167 and 1.99, sd = 0.4
        self.dm = evolvable_params["dm"] * setup.PROPORTION  # move distance, minimum 0, sd = 3
        self.am = evolvable_params["am"]  # move angle, between 0 and 360, sd = 72
        # foraging
        self.df = evolvable_params["df"]  * setup.PROPORTION  # search radius of forager
        self.af = evolvable_params["af"]  # search angle, angle between food and forward direction
        self.tf = evolvable_params["tf"]  # foodscan duration
        self.neighbours = []
        # self.set_initial_evolvable_parameters()

    # SET EVOLVABLE PARAMS

    # def set_repulsion_zone(self, repulsion):
    #     self.zr = repulsion

    # def set_attraction_zone(self, attraction):
    #     self.za = attraction

    # def set_repulsion_angle(self, repulsion):
    #     self.ar = repulsion

    # def set_attraction_angle(self, attraction):
    #     self.aa = attraction

    # sets the initial values of evolvable parameters of the prey agent
    # TODO set all the parameters, currently only grouping is done
    def set_initial_evolvable_parameters(self):

        self.zr = trunc_normal(0, 50, 10)  # random value between 0 and 50, sd = 10
        za = trunc_normal(self.zr, 50, 10)  # random value between self.zr and 50, sd = 10
        self.set_attraction_zone(za)
        aa = trunc_normal(0, 360, 72)  # random value between 0 and 360, sd = 72
        self.set_attraction_angle(aa)
        ar = trunc_normal(0, 360, 72)  # random value between 0 and 360, sd = 72
        self.set_repulsion_angle(ar)


    # STEP FUNCTION
    def step(self):
        self.age = self.age + 1

        self.energy = self.energy - self.em
        print("self energy step1 ", self.energy)

        # Waiting time (after fleeing from predator)
        if self.is_safe == True:
            self.waiting_time = self.waiting_time - 1
            if self.waiting_time == 0:
                self.state = Prey_State.NOTHING
                # reset waiting time
                self.waiting_time = 10
            if getattr(self.model, 'num_prey_agents') > 10 and self.energy >= self.max_energy:
                self.reproduce()
            else:
                self.force_birth()

        # TODO add reproduction and death? (page 9 paper)
        # TODO read closely 1.7.1, should this be done in model? ("model updating schedule")

        # Check if current action is over
        if self.current_action_time_remaining == 0:
            self.previous_state = self.state
            self.state = Prey_State.NOTHING
            self.current_action_time_remaining = self.waiting_time
        else:  # count down time remaining in action by 1
            self.current_action_time_remaining -= 1

        # flee takes precedence over everything, cutting short the current action
        if self.detected_predator != False:
            self.state = Prey_State.FLEEING
            self.flee()
            self.move()

        self.check_group()
            # elif neighbour.get_type() == "predator":



        # complete current action
        # print("Self state prey is ", self.state)
        if self.state == Prey_State.DEAD:
            return
        elif self.state == Prey_State.MOVING:
            self.move()
        elif self.state == Prey_State.FOODSCAN:
            self.food_target = self.foodscan()
            if self.food_target == None:
                self.move()
        elif self.state == Prey_State.MOVETOFOOD:
            self.move_to_food(self.food_target)
        elif self.state == Prey_State.EATING:
            self.energy += self.er
            print("self energy step eating ", self.energy)
            self.food_target = None
        elif self.state == Prey_State.SCANNING:
            # print("SCAN1")
            self.scan()
        elif self.state == Prey_State.FLEEING and (self.is_safe is False):
            self.flee()
            self.move()
        # if current action complete or NONE, choose new action
        elif self.state == Prey_State.NOTHING:
            RAND = np.random.rand()

            # if RAND < self.pv or self.is_safe is True:
            if RAND < self.pv:
                # print("RAND IS ", RAND, " and self.pv ", self.pv)
                # print("is_safe ", self.is_safe)
                self.state = Prey_State.SCANNING
                # print(self.state , " is now ")
            else:
                # print("else pv is ", self.pv)
                if self.food_target is not None:
                    # print("FOOD TARGET")
                    if self.distance(self.food_target) < self.dr:
                        self.state = Prey_State.EATING
                        # print(self.state)
                    else:
                        self.state = Prey_State.MOVETOFOOD
                        # print(self.state)
                else:
                    if self.previous_state == Prey_State.MOVING:
                        if RAND < self.pm:
                            self.state = Prey_State.MOVING
                            # print(self.state)
                        else:
                            self.state = Prey_State.FOODSCAN
                            # print(self.state)
                    elif self.previous_state == Prey_State.EATING:
                        if RAND < self.pse:
                            self.state = Prey_State.FOODSCAN
                            # print(self.state)
                        else:
                            self.state = Prey_State.MOVING
                            # print(self.state)
                    elif self.previous_state == Prey_State.FLEEING:
                        self.state = Prey_State.SCANNING
                    elif self.previous_state == Prey_State.SCANNING:
                        if RAND < self.pse:
                            self.state = Prey_State.FOODSCAN
                            # print(self.state)
                        else:
                            self.state = Prey_State.MOVING
                            # print(self.state)
                    elif self.previous_state == Prey_State.NOTHING:
                        if RAND < self.pse:
                            self.state = Prey_State.FOODSCAN
                            # print(self.state)
                        else:
                            self.state = Prey_State.MOVING
                            # print(self.state)
        # print("final state is ", self.state)

    # TODO choose random parent, force birth with no energy cost --> possibly actually should do that in model?

    # PREY ACTIONS
    # o = facing N, 90  facing right/E, 180 facing S, 270 = facing L/W
    def define_angle_space(self, di):
        if di >= 0 and di <= 90:
            pos =( self.pos[0] + 1, self.pos[1] + 1)
            return pos
        if di >= 90 and di <= 180:
            pos =( self.pos[0] + 1, self.pos[1])
            return pos
        if di >= 180 and di <= 270:
            pos =( self.pos[0] , self.pos[1] - 1)
            return pos
        if di >= 270 and di <= 360:
            pos =( self.pos[0] - 1, self.pos[1] - 1)
            return pos

    def search_space(self):
        new_position = self.define_angle_space(self.di)
        return new_position

    def move(self):
        # Grouping
        # print("PREY IS MOVING TO ")
        # define self.direction of facing the group
        if self.nrz >= self.nr: # repulsion
            if self.ar < self.di:
                self.di = self.ar
        else: # attraction
            if self.aa < self.di:
                self.di = self.aa
        new_position = self.search_space()
        # print(new_position)
        self.model.grid.move_agent(self, new_position)
        self.set_position(new_position)

        self.current_action_time_remaining = self.dm * self.tm

    def new_move(self):
        # TODO how to get neighbours, but only of class prey, current idea isn't efficent
        # Grouping params
        # get number of actual neighbors within zones
        count_neighbours_repulsed = 0
        for x in self.model.grid.get_neighbors(self.get_position(), radius=self.zr):
            if x.type == "prey":
                count_neighbours_repulsed += 1
        nrz = count_neighbours_repulsed # actual neighbours in repulsion zone

        count_neighbours = 0
        for x in self.model.grid.get_neighbors(self.get_position(), radius=self.zl):
            if x.type == "prey":
                count_neighbours += 1
        nl = count_neighbours - count_neighbours_repulsed # actual neighbours in alignment zone

        count_neighbours = 0
        for x in self.model.grid.get_neighbors(self.get_position(), radius=self.za):
            if x.type == "prey":
                count_neighbours += 1
        na = count_neighbours - count_neighbours_repulsed # actual neighbours in alignment and attraction zone

        # Grouping
        if nrz >= self.nr:
            sum = 0
            for x in self.model.grid.get_neighbors(self.get_position(), radius=self.zr):
                if x.type == "prey":
                    sum += (self.get_position() - x.get_position()) / abs(self.get_position() - x.get_position())
            d_hat = - sum / abs(sum)

        else:
            sum1 = 0
            for x in self.model.grid.get_neighbors(self.get_position(), radius=self.za):
                if x.type == "prey":
                    sum1 += (self.get_position() - x.get_position()) / abs(self.get_position() - x.get_position())
            sum2 = 0
            for x in self.model.grid.get_neighbors(self.get_position(), radius=self.zl):
                if x.type == "prey":
                    sum2 += x.v_hat
            d_hat = (sum1 + sum2) / abs(sum1 + sum2)
        if abs(self.di - d) <= self.ar or abs(self.di - d) <= self.aa:
            self.v_hat = d_hat
        else:
            vx = self.v_hat[1]
            vy = self.v_hat[2]
            # TODO it says "else turn aR or aA" but I'm not sure how to know which, so currently random?
            if random.random() < 0.5:
                a = self.ar
            else:
                a = self.aa
            vx = vx * math.cos(a) - vy * math.sin(a)
            vy = vx * math.cos(a) + vy * math.sin(a)
            self.v_hat = [vx, vy]

        # random turn of a_M
        if random.random() < 0.5:
            t = - self.am
        else:
            t = self.am
        vx = self.v_hat[1]
        vy = self.v_hat[2]
        vx = vx * math.cos(t) - vy * math.sin(t)
        vy = vx * math.cos(t) + vy * math.sin(t)
        self.v_hat = [vx, vy]

        # Get new position and make sure it is on the grid
        new_position = self.dm * self.v_hat + self.position

        # Set new pos
        self.position = new_position

        # Duration
        self.current_action_time_remaining = self.dm * self.tm

    def check_group(self):
        for neighbour in self.model.grid.get_neighbors(self.position, moore=True, include_center=False,
                                                       radius=self.max_neighbour_awareness):
            if neighbour.get_type() == "prey":
                self.nrz += 1
                self.di = ( self.di + neighbour.di ) / 2
                if neighbour.get_state() == Prey_State.FLEEING:
                    self.state = Prey_State.FLEEING
                    self.flee()
                    self.move()

    def distance(self, otherpos):
        # print(self.pos , " is th eposition ")
        # (distance_x, distance_y) = (self.pos[0] - otherpos[0], self.pos[1] - otherpos[1])
        dist = np.sum(np.square(np.array((self.pos)), np.array((otherpos))))
        # print("distance is ", np.sqrt(dist))
        return np.sqrt(dist)

    # TODO: should this return chosen fooditem or set a field to this fooditem
    def foodscan(self):
        # chosenitem = self.model.fooditems[0]

        # for neighbour in self.model.grid.get_neighbors(self.position, moore=True, include_center=False,
        #                                                radius=self.max_neighbour_awareness):
        #     if neighbour.get_type() == "predator":
        #         predator_distance = self.distance(neighbour.get_position())
        #         pd = pow(self.h, self.N) / ((pow(predator_distance, self.N)) * pow(self.h, self.N)) * (
        #                     math.pi / self.av) * (self.tv / self.t_min)
        #         if pd < random.random():
        #             self.detected_predator = neighbour
        #             break
        #
        # if self.detected_predator is None:
        #     self.current_action_time_remaining = self.tv
        # else:
        #     self.current_action_time_remaining = random.randint(0, self.tv)

        chosenitems = self.model.grid.get_neighbors(self.position, moore=True, include_center=False,
                                                      radius=self.max_neighbour_awareness)
        chosenitem = None
        for chosenitem in chosenitems:
            if chosenitem.type == "food":
                chosenitem = chosenitem
                break
        # find all fooditems in range
        # for fooditem in range(len(self.model.fooditems)):
        #     p = (self.tf * 60) / (
        #             np.pi * pow(self.df, 2) * (self.af / np.pi))  # we assume this function represents the vision
        #     RAND = random.random()
        #     if RAND < p:
        #         if self.distance(fooditem.pos) < self.distance(chosenitem.pos):
        #             chosenitem = fooditem

        return chosenitem

    # TODO related to foodscan, should this have a fooditem as argument or should it move to variable "self.closestfood" or st
    def move_to_food(self, food_item):
        new_position = food_item.position - \
                       (self.dr * abs(food_item.position - self.position)) / 2
        self.position = new_position
        self.current_action_time_remaining = self.distance(new_position)
        self.move()

    def eat(self, food_item):
        # resource items that are eaten disappear immediately (no half eating possible)
        self.model.remove_agents_food.append(food_item)
        # remove the agent from the grid, immediately to prevent it being eaten twice
        food_item.remove_agent()
        self.current_action_time_remaining = self.current_action_time_remaining - self.te



    def scan(self):
        for neighbour in self.model.grid.get_neighbors(self.position, moore=True, include_center=False,
                                                       radius=self.max_neighbour_awareness):
            if neighbour.get_type() == "predator":
                print("detect predator, with h ", self.h)
                predator_distance = self.distance(neighbour.position)
                pd = pow(self.h, self.N) / ((pow(predator_distance, self.N)) * pow(self.h, self.N)) * (
                            math.pi / self.av) * (self.tv / self.t_min)
                if pd < random.random():
                    print("pd is ", pd)
                    self.detected_predator = neighbour
                    break

        if self.detected_predator is None:
            self.current_action_time_remaining = self.tv
        else:
            self.current_action_time_remaining = random.randint(0, int(self.tp))

    def flee(self):
        # No change in spatial position, safety is simply assumed
        self.is_safe = True
        self.detected_predator = None
        self.current_action_time_remaining = self.current_action_time_remaining - self.reaction_time

    def reproduce(self):
        # Reproduction
        # neighbours = self.model.grid.get_neighbors(self.position, include_center=True)
        # n_children = int((len(neighbours)/2))
        # self.model.create_prey(n_children)
        # energy changes due to birth
        self.energy = self.energy - self.em / 2
        print("self energy step1 ", self.energy)
        # TODO params is the baseclass name maybe use evolvable
        child_params = mutate(deepcopy(self.evolvable_params))
        print("child params ", child_params)
        self.model.create_new_prey(child_params)
        print("created new prey!!!")
        # a = PreyAgent(getattr(self.model, 'num_prey_agents') + 1, self.model, child_params)
        # TODO offspring inherit all evolvable parameters + mutate, maybe make functions inherit() and evolve()


    def force_birth(self):
        n = 5
        print("self energy step force birth ", self.energy)
        summed_energy_neighbours = 0
        # for agent in self.model.grid.get_neighbors((setup.GRID_WIDTH, setup.GRID_HEIGHT),
        #                                             moore=True, include_center=True,
        #                                             radius=self.max_neighbour_awareness):
        #     print("agent energy ", agent.energy)
        #     summed_energy_neighbours += agent.energy
        # summed_energy_neighbours = max(summed_energy_neighbours, 1)
        # prob_to_birth = math.pow(self.energy / summed_energy_neighbours, n)
        # print(" prob to birth ", str(prob_to_birth))
        # if np.random.random() > prob_to_birth:
        if np.random.random() > 0.5:
            self.reproduce()

    def set_energy(self, new_energy):
        self.energy = new_energy

    def get_state(self):
        return self.state

    def is_alive(self):
        return self.state != Prey_State.DEAD

        