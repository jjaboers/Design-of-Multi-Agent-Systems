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

import prey_params


class Prey_State(Enum):
    NOTHING = 1  # TODO called "NORMAL" in paper?
    MOVING = 3
    FOODSCAN = 4
    MOVETOFOOD = 5
    EATING = 6
    SCANNING = 7
    FLEEING = 8
    DEAD = 9

# Truncated normal distribution, takes range [lower, upper] and standard deviation (sd)


def trunc_normal(lower, upper, sd, mu):
    mu = upper - lower
    r = truncnorm.rvs(
        (lower - mu) / sd, (upper - mu) / sd, loc=mu, scale=sd, size=1)
    return r


def mutate(params):
    #print("params are " , params)
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


class PreyAgent(TypedAgent):
    """An agent that is a prey, as described in the paper."""

    def __init__(self, unique_id, model, default_params=prey_params.default_params_prey, evolvable_params=prey_params.evolvable_params_prey):
        super().__init__(unique_id, model)
        self.type = "prey"
        # self.model = model
        self.state = Prey_State.NOTHING
        self.previous_state = Prey_State.NOTHING
        self.current_action_time_remaining = 0
        self.detected_predator = False  # keep it like this or make it Boolean ?
        self.age = 0 # Was 100
        self.energy = 10 #Was 100000
        self.min_energy = 0
        self.default_params = default_params
        self.evolvable_params = evolvable_params

        self.position = default_params["position"]
        self.food_target = default_params["food_target"]

        self.zl = default_params["zl"]
        self.dr = default_params["dr"] * setup.PROPORTION
        self.max_speed = default_params["max_speed"] * setup.PROPORTION
        self.max_neighbour_awareness = default_params["max_neighbour_awareness"] * \
            setup.PROPORTION if default_params["max_neighbour_awareness"] * \
            setup.PROPORTION > 1 else 1
        self.h = default_params["h"] if default_params["h"] > 5 else 5
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
        self.v_hat = default_params["v_hat"]

        # decision making
        # predator scan, between 0 and 1, sd = 0.2
        self.pv = evolvable_params["pv"]
        # move after move, between 0 and 1, sd = 0.2
        self.pm = evolvable_params["pm"]
        # food scan after eat, between 0 and 1, sd = 0.2
        self.pse = evolvable_params["pse"]
        # food scan after no food, between 0 and 1, sd = 0.2
        self.psn = evolvable_params["psn"]
        # move to food, between 0 and 1, sd = 0.2
        self.pmtf = evolvable_params["pmtf"]
        # vigilance
        # scan duration, between 0.167 and 1.99, sd = 0.4
        self.tv = evolvable_params["tv"]
        # scan angle, between 0 and 360, sd = 72
        self.av = evolvable_params["av"]
        # fleeing
        self.tp = evolvable_params["tp"]  # flee duration, minimum 0, sd = 5
        # grouping
        # repulsion zone, between 0 and 50, sd = 10
        self.zr = evolvable_params["zr"]
        # attraction zone, between zr and 50, sd = 10
        self.za = evolvable_params["za"]
        # maximum turning angle for attraction, between 0 and 360, sd = 72
        self.aa = evolvable_params["aa"]
        # maximum turning angle for repulsion, between 0 and 360, sd = 72
        self.ar = evolvable_params["ar"]
        self.nr = evolvable_params["nr"]  # tolerated neighbors, (0 or 1)
        # movement
        # move duration, between 0.167 and 1.99, sd = 0.4
        self.tm = evolvable_params["tm"]
        # move distance, minimum 0, sd = 3
        self.dm = evolvable_params["dm"] * setup.PROPORTION
        # move angle, between 0 and 360, sd = 72
        self.am = evolvable_params["am"]
        # foraging
        self.df = evolvable_params["df"] * \
            setup.PROPORTION  # search radius of forager
        # search angle, angle between food and forward direction
        self.af = evolvable_params["af"]
        self.tf = evolvable_params["tf"]  # foodscan duration
        self.neighbours = []
        # self.set_initial_evolvable_parameters()


    # sets the initial values of evolvable parameters of the prey agent
    # TODO set all the parameters, currently only grouping is done
    def set_initial_evolvable_parameters(self):

        # random value between 0 and 50, sd = 10
        self.zr = trunc_normal(0, 50, 10)
        # random value between self.zr and 50, sd = 10
        za = trunc_normal(self.zr, 50, 10)
        self.set_attraction_zone(za)
        # random value between 0 and 360, sd = 72
        aa = trunc_normal(0, 360, 72)
        self.set_attraction_angle(aa)
        # random value between 0 and 360, sd = 72
        ar = trunc_normal(0, 360, 72)
        self.set_repulsion_angle(ar)

    # STEP FUNCTION
    def step(self):
        print("STATE")
        print(self.state)
        # print("time remaining:", self.current_action_time_remaining)
        self.age = self.age + 1

        self.energy = self.energy - self.em
        #print("self energy step1 ", self.energy)

        # Waiting time (after fleeing from predator)
        if self.is_safe == True:
            self.waiting_time = self.waiting_time - 1
            # print("I'm safe bestie")
            # print(getattr(self.model, 'num_prey_agents'))
            if self.waiting_time == 0:
                self.state = Prey_State.NOTHING
                # reset waiting time
                self.waiting_time = 10
            if getattr(self.model, 'num_prey_agents') > 10 and self.energy >= self.max_energy:
                # print("I can reproduce")
                self.reproduce()
            elif getattr(self.model, 'num_prey_agents') < 10:
                # print("Gotta force birth")
                self.force_birth()

        # TODO add reproduction and death? (page 9 paper)
        # TODO read closely 1.7.1, should this be done in model? ("model updating schedule")

        # Check if current action is over
        if self.current_action_time_remaining < 0:
            self.previous_state = self.state
            self.state = Prey_State.NOTHING
            self.current_action_time_remaining = self.waiting_time
        else:  # count down time remaining in action by 1
            self.current_action_time_remaining -= 1

        # flee takes precedence over everything, cutting short the current action
        if self.detected_predator != False:
            self.state = Prey_State.FLEEING
            self.flee()
            self.new_move()

        self.check_group()
        # elif neighbour.get_type() == "predator":

        # complete current action
        # print("Self state prey is ", self.state)
        if self.state == Prey_State.DEAD:
            return
        elif self.state == Prey_State.MOVING:
            self.new_move()
        elif self.state == Prey_State.FOODSCAN:
            self.food_target = self.foodscan()
            # print(self.food_target)
            if self.food_target == None:
                self.new_move()
        elif self.state == Prey_State.MOVETOFOOD: # braek out of movetofood if at food
            if self.distance(self.food_target.position) <= self.dr:
                # print("distance", self.distance(self.food_target.position), self.dr)
                self.state = Prey_State.EATING
                self.current_action_time_remaining = self.te
            else:
                # print("trying to move to food")
                self.move_to_food(self.food_target)
        elif self.state == Prey_State.EATING:
            # print("am eating")
            if self.food_target != None:
                self.energy += self.er
                self.eat(self.food_target)
                # print("self energy step eating ", self.energy)
                self.food_target = None
        elif self.state == Prey_State.SCANNING:
            # print("SCAN1")
            self.scan()
        elif self.state == Prey_State.FLEEING and (self.is_safe is False):
            self.flee()
            self.new_move()
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
                    # print("FOOD TARGET", self.food_target)
                    if self.distance(self.food_target.position) <= self.dr:
                        # print("distance", self.distance(self.food_target.position), self.dr)
                        self.state = Prey_State.EATING
                        self.current_action_time_remaining = self.te
                        # print(self.state)
                    else:
                        self.state = Prey_State.MOVETOFOOD
                        # print(self.state)
                else:
                    if self.previous_state == Prey_State.MOVING:
                        if RAND < self.pm:
                            # print("RAND IS ", RAND, " and self.pm ", self.pm)
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
            pos = (self.pos[0] + 1, self.pos[1] + 1)
            return pos
        if di >= 90 and di <= 180:
            pos = (self.pos[0] + 1, self.pos[1])
            return pos
        if di >= 180 and di <= 270:
            pos = (self.pos[0], self.pos[1] - 1)
            return pos
        if di >= 270 and di <= 360:
            pos = (self.pos[0] - 1, self.pos[1] - 1)
            return pos

    def search_space(self):
        new_position = self.define_angle_space(self.di)
        return new_position

    def move(self):
        # Grouping
        # print("PREY IS MOVING TO ")
        # define self.direction of facing the group
        if self.nrz >= self.nr:  # repulsion
            if self.ar < self.di:
                self.di = self.ar
        else:  # attraction
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
        d_hat = np.array([0, 0])
        count_neighbors = 0
        count_neighbours_repulsed = 0
        current_position = np.array([self.position[0], self.position[1]])
        pos = (self.position[0], self.position[0])
        for x in self.model.grid.get_neighbors(pos, radius=self.zr, include_center=False):
            if x.type == "prey":
                count_neighbours_repulsed += 1
        nrz = count_neighbours_repulsed  # actual neighbours in repulsion zone

        count_neighbours = 0
        for x in self.model.grid.get_neighbors(pos, radius=self.zl, include_center=False):
            if x.type == "prey":
                count_neighbours += 1
        # actual neighbours in alignment zone
        nl = count_neighbours - count_neighbours_repulsed

        count_neighbours = 0
        for x in self.model.grid.get_neighbors(pos, radius=self.za, include_center=False):
            if x.type == "prey":
                count_neighbours += 1
        # actual neighbours in alignment and attraction zone
        na = count_neighbours - count_neighbours_repulsed

        # Grouping
        if nrz >= self.nr:
            sum0 = np.array([0, 0])
            dist = 0
            abs_dist = 0
            for x in self.model.grid.get_neighbors(pos, radius=self.zr, include_center=False):
                if x.type == "prey":
                    x_position = np.array([x.position[0], x.position[1]])
                    dist = x_position - current_position
                    abs_dist = math.sqrt(
                        (dist[0] * dist[0]) + (dist[1] * dist[1]))
                    if abs_dist != 0:
                        sum0 = sum0 + (dist / abs_dist)
            abs_sum = math.sqrt((sum0[0] * sum0[0]) + (sum0[1] * sum0[1]))
            d_hat = - sum0 / abs_sum

        else:
            sum1 = np.array([0, 0])
            dist = 0
            abs_dist = 0
            for x in self.model.grid.get_neighbors(pos, radius=self.za, include_center=False):
                if x.type == "prey":
                    x_position = np.array([x.position[0], x.position[1]])
                    # sum1 += (current_position- x_position) / abs(current_position - x_position)
                    dist = current_position - x_position
                    abs_dist = math.sqrt(
                        (dist[0] * dist[0]) + (dist[1] * dist[1]))
                    if abs_dist != 0:
                        sum1 = sum1 + (dist / abs_dist)
            sum2 = np.array([0, 0])
            for x in self.model.grid.get_neighbors(pos, radius=self.zl,  include_center=False):
                if x.type == "prey":
                    sum2 = np.array(x.v_hat) + sum2
            sums = sum1 + sum2
            abs_sums = math.sqrt((sums[0] * sums[0]) + (sums[1] * sums[1]))
            d_hat = - sums / abs_sums
            # d_hat = (sum1 + sum2) / abs(sum1 + sum2)

        # calculate angle between v_hat and d_hat
        dot_product = (self.v_hat[0] * d_hat[0]) + (self.v_hat[1] * d_hat[1])
        v_abs = np.sqrt(
            (self.v_hat[0] * self.v_hat[0]) + (self.v_hat[1] * self.v_hat[1]))
        d_abs = np.sqrt((d_hat[0] * d_hat[0]) + (d_hat[1] * d_hat[1]))
        angle = math.acos(dot_product / v_abs * d_abs)
        # convert to degrees, ensure positive
        angle = abs(angle * (180.0 / math.pi))

        if angle <= self.ar or angle <= self.aa:
            self.v_hat = d_hat
        else:
            vx = self.v_hat[0]
            vy = self.v_hat[1]
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
        vx = self.v_hat[0]
        vy = self.v_hat[1]
        vx = vx * math.cos(t) - vy * math.sin(t)
        vy = vx * math.cos(t) + vy * math.sin(t)
        self.v_hat = [vx, vy]

        # Get new position and make sure it is on the grid
        new_position = self.dm * self.v_hat + self.position
        # TODO Rounding is causing problems but not rounding causes issues in the get_neighbour calls
        new_position_rounded = (new_position[0], self.position[1])
        # Set new pos
        self.model.grid.move_agent(self, new_position_rounded)
        self.position = (tuple(new_position_rounded))
        # Duration
        self.current_action_time_remaining = self.dm * self.tm

    def check_group(self):
        pos = (self.position[0], self.position[0])
        for neighbour in self.model.grid.get_neighbors(pos,  include_center=False,
                                                       radius=self.max_neighbour_awareness):
            if neighbour.get_type() == "prey":
                self.nrz += 1
                self.di = (self.di + neighbour.di) / 2
                if neighbour.get_state() == Prey_State.FLEEING:
                    self.state = Prey_State.FLEEING
                    self.flee()
                    self.new_move()

    def distance(self, otherpos):
        # print(self.pos, otherpos, " is the position ")
        # (distance_x, distance_y) = (self.pos[0] - otherpos[0], self.pos[1] - otherpos[1])
        dist = pow((self.pos[0] - otherpos[0]),2) + pow((self.pos[1] - otherpos[1]),2)
        # dist = np.sum(np.square(np.array((self.pos)), np.square(np.array((otherpos)))))
        # print("distance is ", np.sqrt(dist))
        return np.sqrt(dist)

    # TODO: should this return chosen fooditem or set a field to this fooditem
    def foodscan(self):
        # print("in foodscan")
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

        chosenitems = self.model.grid.get_neighbors(self.position, include_center=False,
                                                    radius=self.max_neighbour_awareness)
        # print("max neigh", self.max_neighbour_awareness)
        chosenitem_ = None

        for chosenitem in chosenitems:
            # print("prey pos", self.position)
            # print(chosenitem.position)
            if chosenitem.type == "food":
                chosenitem_ = chosenitem
                break
        # find all fooditems in range
        # for fooditem in range(len(self.model.fooditems)):
        #     p = (self.tf * 60) / (
        #             np.pi * pow(self.df, 2) * (self.af / np.pi))  # we assume this function represents the vision
        #     RAND = random.random()
        #     if RAND < p:
        #         if self.distance(fooditem.pos) < self.distance(chosenitem.pos):
        #             chosenitem = fooditem

        return chosenitem_

    # TODO related to foodscan, should this have a fooditem as argument or should it move to variable "self.closestfood" or st
    def move_to_food(self, food_item):
        # new_position = food_item.position - \
        # (self.dr * abs(food_item.position - self.position)) / 2
        # dif = (abs(food_item.position[0] - self.position[0] / 2), abs(food_item.position[1] - self.position[1] / 2))
        # new_position = food_item.position - (self.dr * dif)
        '''
        x = food_item.position[0] - self.dr * \
            abs(food_item.position[0] - self.position[0] / 2)
        y = food_item.position[1] - self.dr * \
            abs(food_item.position[1] - self.position[1] / 2)
        '''
        x = food_item.position[0] - (self.dr/2) * abs(food_item.position[0] - self.position[0])
        y = food_item.position[1] - (self.dr/2) * abs(food_item.position[1] - self.position[1])
        new_position = (x, y)
        # print("Self:", self.position, "Food", food_item.position, "New:", new_position)
        self.position = new_position
        self.current_action_time_remaining = self.distance(new_position)
        self.new_move()

    def eat(self, food_item):
        # print("Nom")
        # resource items that are eaten disappear immediately (no half eating possible)
        self.model.remove_agents_food.append(food_item)
        # remove the agent from the grid, immediately to prevent it being eaten twice
        self.model.grid.remove_agent(food_item)
        #self.current_action_time_remaining = self.current_action_time_remaining - self.te

    def scan(self):
        # print(self.position)
        for neighbour in self.model.grid.get_neighbors(self.position, include_center=False,
                                                       radius=self.max_neighbour_awareness):
            if neighbour.get_type() == "predator":
                #("detect predator, with h ", self.h)
                predator_distance = self.distance(neighbour.position)
                pd = pow(self.h, self.N) / ((pow(predator_distance, self.N)) * pow(self.h, self.N)) * (
                    math.pi / self.av) * (self.tv / self.t_min)
                if pd < random.random():
                    #print("pd is ", pd)
                    self.detected_predator = neighbour
                    break

        if self.detected_predator is None:
            self.current_action_time_remaining = self.tv
        else:
            self.current_action_time_remaining = np.random.random() * self.tp + 1

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
        self.energy = self.energy - self.max_energy / 2
        #print("self energy step1 ", self.energy)
        # TODO params is the baseclass name maybe use evolvable
        child_params = mutate(deepcopy(self.evolvable_params))
        #print("child params ", child_params)
        self.model.create_new_prey(child_params)
        #print("created new prey!!!")
        # a = PreyAgent(getattr(self.model, 'num_prey_agents') + 1, self.model, child_params)
        # TODO offspring inherit all evolvable parameters + mutate, maybe make functions inherit() and evolve()

    # def force_birth(self):
    #     n = 5
    #     #print("self energy step force birth ", self.energy)
    #     summed_energy_neighbours = 0
    #     # for agent in self.model.grid.get_neighbors((setup.GRID_WIDTH, setup.GRID_HEIGHT),
    #     #                                             moore=True, include_center=True,
    #     #                                             radius=self.max_neighbour_awareness):
    #     #     print("agent energy ", agent.energy)
    #     #     summed_energy_neighbours += agent.energy
    #     # summed_energy_neighbours = max(summed_energy_neighbours, 1)
    #     # prob_to_birth = math.pow(self.energy / summed_energy_neighbours, n)
    #     # print(" prob to birth ", str(prob_to_birth))
    #     # if np.random.random() > prob_to_birth:
    #     if np.random.random() > 0.5:
    #         self.reproduce()

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

    def set_energy(self, new_energy):
        self.energy = new_energy

    def get_state(self):
        return self.state

    def is_alive(self):
        return self.state != Prey_State.DEAD
