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
        self.state = Prey_State.NOTHING
        self.previous_state = Prey_State.NOTHING
        self.current_action_time_remaining = 0
        self.detected_predator = False  # keep it like this or make it Boolean ?
        self.max_age = default_params["max_age"] * setup.PROPORTION
        self.age = self.max_age
        self.energy = 100000 * setup.PROPORTION

        self.min_energy = 0
        self.default_params = default_params
        self.evolvable_params = evolvable_params

        self.position = default_params["position"]
        self.food_target = default_params["food_target"]

        self.zl = default_params["zl"]
        self.dr = default_params["dr"]
        self.max_speed = default_params["max_speed"]
        self.max_neighbour_awareness = default_params["max_neighbour_awareness"]

        self.h = default_params["h"] * \
            setup.PROPORTION if default_params["h"] > 5 else 5
        self.N = default_params["N"]
        self.em = default_params["em"]
        self.max_energy = 100000 * setup.PROPORTION
        self.death_rate = default_params["death_rate"]
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
        self.zr = evolvable_params["zr"] * setup.PROPORTION
        # attraction zone, between zr and 50, sd = 10
        self.za = evolvable_params["za"] * setup.PROPORTION
        # maximum turning angle for attraction, between 0 and 360, sd = 72
        self.aa = evolvable_params["aa"]
        # maximum turning angle for repulsion, between 0 and 360, sd = 72
        self.ar = evolvable_params["ar"]
        self.nr = evolvable_params["nr"]  # tolerated neighbors, (0 or 1)
        # movement
        # move duration, between 0.167 and 1.99, sd = 0.4
        self.tm = evolvable_params["tm"]
        # move distance, minimum 0, sd = 3
        self.dm = evolvable_params["dm"] * setup.PROPORTION * 10
        # move angle, between 0 and 360, sd = 72
        self.am = evolvable_params["am"]
        # foraging
        self.df = evolvable_params["df"]
        # search radius of forager
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
        self.age = self.age + 1
        self.nrz = 0
        self.energy = self.energy - self.em

        if self.energy <= self.min_energy:
            self.state = Prey_State.DEAD

        if self.age <= 0:
            self.state = Prey_State.DEAD

        if self.state == Prey_State.DEAD:
            self.die()

        # Waiting time (after fleeing from predator)
        if self.is_safe == True:
            self.waiting_time = self.waiting_time - 1
            if self.waiting_time == 0:
                self.state = Prey_State.NOTHING
                self.is_safe = False
                self.waiting_time = 10
            if getattr(self.model, 'num_prey_agents') > 10 and self.energy >= self.max_energy:
                self.reproduce()
            elif getattr(self.model, 'num_prey_agents') < 10:
                self.force_birth()


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

        # complete current action
        if self.state == Prey_State.MOVING:
            self.new_move()
        elif self.state == Prey_State.FOODSCAN:
            self.food_target = self.foodscan()
            if self.food_target == None:
                self.new_move()
        elif self.state == Prey_State.MOVETOFOOD:  # braek out of movetofood if at food
            if self.distance(self.food_target.position) <= self.dr:
                self.state = Prey_State.EATING
                self.current_action_time_remaining = self.te
            else:
                self.move_to_food(self.food_target)
        elif self.state == Prey_State.EATING:
            if self.food_target != None:
                self.energy += self.er
                self.eat(self.food_target)
                self.food_target = None
        elif self.state == Prey_State.SCANNING:
            self.scan()
        elif self.state == Prey_State.FLEEING and (self.is_safe is False):
            self.flee()
            self.new_move()
        # if current action complete or NONE, choose new action
        elif self.state == Prey_State.NOTHING:
            RAND = np.random.rand()

            if RAND < self.pv:
                self.state = Prey_State.SCANNING
            else:
                if self.food_target is not None:
                   
                    if self.distance(self.food_target.position) <= self.dr:
                    
                        self.state = Prey_State.EATING
                        self.current_action_time_remaining = self.te
                    else:
                        self.state = Prey_State.MOVETOFOOD
                else:
                    if self.previous_state == Prey_State.MOVING:
                        RAND = np.random.rand()
                        if RAND < self.pm:
                            self.state = Prey_State.MOVING
                        else:
                            self.state = Prey_State.FOODSCAN
                    elif self.previous_state == Prey_State.EATING:
                        RAND = np.random.rand()
                        if RAND < self.pse:
                            self.state = Prey_State.FOODSCAN
                        else:
                            self.state = Prey_State.MOVING
                    elif self.previous_state == Prey_State.FLEEING:
                        self.state = Prey_State.SCANNING
                    elif self.previous_state == Prey_State.SCANNING:
                        RAND = np.random.rand()
                        if RAND < self.pse:
                            self.state = Prey_State.FOODSCAN
                        else:
                            self.state = Prey_State.MOVING
                    elif self.previous_state == Prey_State.NOTHING:
                        RAND = np.random.rand()
                        if RAND < self.pse:
                            self.state = Prey_State.FOODSCAN
                        else:
                            self.state = Prey_State.MOVING



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
        # define self.direction of facing the group
        if self.nrz >= self.nr:  # repulsion
            if self.ar < self.di:
                self.di = self.ar
        else:  # attraction
            if self.aa < self.di:
                self.di = self.aa
        new_position = self.search_space()
        self.model.grid.move_agent(self, new_position)
        self.set_position(new_position)

        self.current_action_time_remaining = self.dm * self.tm

    def new_move(self):
        # Grouping params
        # get number of actual neighbors within zones
        d_hat = np.array([0, 0])
        count_neighbors = 0
        count_neighbours_repulsed = 0
        current_position = np.array([self.position[0], self.position[1]])
        pos = (self.position[0], self.position[0])
        max_radius = max(self.zr, self.zl, self.za)

        zr_agents = []
        za_agents = []
        zl_agents = []
        nrz = 0
  
        for x in self.model.grid.get_neighbors(pos, radius=max_radius, include_center=False):
            if x.type == "prey":
                distance = self.distance(x.position)
                if distance <= self.zr:
                    count_neighbours_repulsed += 1
                    zr_agents.append(x)
            nrz = count_neighbours_repulsed  # actual neighbours in repulsion zone
            if x.type == "prey":
                distance = self.distance(x.position)
                if distance <= self.zl:
                    zl_agents.append(x)
     
            if x.type == "prey":
                distance = self.distance(x.position)
                if distance <= self.za:
                    za_agents.append(x)
            
        # Grouping
        if nrz >= self.nr:
            sum0 = np.array([0, 0])
            dist = 0
            abs_dist = 0
            for x in zr_agents:
                x_position = np.array([x.position[0], x.position[1]])
                dist = x_position - current_position
                abs_dist = math.sqrt(
                    (dist[0] * dist[0]) + (dist[1] * dist[1]))
                if abs_dist != 0:
                    sum0 = sum0 + (dist / abs_dist)
            abs_sum = math.sqrt((sum0[0] * sum0[0]) + (sum0[1] * sum0[1]))
            if abs_sum != 0:
                d_hat = - sum0 / abs_sum
            else:
                d_hat = - sum0

        else:
            sum1 = np.array([0, 0])
            dist = 0
            abs_dist = 0
            for x in za_agents:
                x_position = np.array([x.position[0], x.position[1]])
                dist = current_position - x_position
                abs_dist = math.sqrt(
                    (dist[0] * dist[0]) + (dist[1] * dist[1]))
                if abs_dist != 0:
                    sum1 = sum1 + (dist / abs_dist)
            sum2 = np.array([0, 0])
            for x in zl_agents:
                sum2 = np.array(x.v_hat) + sum2
            sums = sum1 + sum2
            abs_sums = math.sqrt((sums[0] * sums[0]) + (sums[1] * sums[1]))
            if abs_sums != 0:
                d_hat = - sums / abs_sums
            else:
                d_hat = np.array([0, 0])

        # calculate angle between v_hat and d_hat
        dot_product = (self.v_hat[0] * d_hat[0]) + (self.v_hat[1] * d_hat[1])
        v_abs = np.sqrt(
            (self.v_hat[0] * self.v_hat[0]) + (self.v_hat[1] * self.v_hat[1]))
        d_abs = np.sqrt((d_hat[0] * d_hat[0]) + (d_hat[1] * d_hat[1]))
        if v_abs == 0.0:
            angle = round(random.uniform(0, math.pi), 2)
        else:
            x = round(dot_product / v_abs * d_abs, 2)
            angle = math.acos(x)
            # convert to degrees, ensure positive
            angle = abs(angle * (180.0 / math.pi))

        if angle <= self.ar or angle <= self.aa:
            self.v_hat = d_hat
        else:
            vx = self.v_hat[0]
            vy = self.v_hat[1]
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
        if (self.v_hat[0] + self.v_hat[1] != 0.0):
            new_position = self.dm * self.v_hat + self.position
        else:
            self.v_hat = np.concatenate((self.pm, self.pm))
            new_position = self.dm * self.v_hat + self.position

        new_position_rounded = new_position
        # Set new pos
        
        if (self.model.grid.out_of_bounds(new_position_rounded)):
            new_position_rounded = self.model.grid.torus_adj(
                new_position_rounded)
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

        dist = pow((self.pos[0] - otherpos[0]), 2) + \
            pow((self.pos[1] - otherpos[1]), 2)

        return np.sqrt(dist)


    def foodscan(self):
        

        chosenitems = self.model.grid.get_neighbors(self.position, include_center=False,
                                                    radius=self.max_neighbour_awareness)
        chosenitem_ = None

        for chosenitem in chosenitems:
            if chosenitem.type == "food":
                chosenitem_ = chosenitem
                break


        return chosenitem_

    def move_to_food(self, food_item):

        '''
        x = food_item.position[0] - self.dr * \
            abs(food_item.position[0] - self.position[0] / 2)
        y = food_item.position[1] - self.dr * \
            abs(food_item.position[1] - self.position[1] / 2)
        '''
        x = food_item.position[0] - (self.dr/2) * \
            abs(food_item.position[0] - self.position[0])
        y = food_item.position[1] - (self.dr/2) * \
            abs(food_item.position[1] - self.position[1])
        new_position = (x, y)
        self.position = new_position
        self.current_action_time_remaining = self.distance(new_position)
        

    def eat(self, food_item):
        
        # resource items that are eaten disappear immediately (no half eating possible)
        self.model.remove_agents_food.append(food_item)
        # remove the agent from the grid, immediately to prevent it being eaten twice
        self.model.grid.remove_agent(food_item)

    def scan(self):
        for neighbour in self.model.grid.get_neighbors(self.position, include_center=False,
                                                       radius=self.max_neighbour_awareness):
            if neighbour.get_type() == "predator":
                predator_distance = self.distance(neighbour.position)
                pd = pow(self.h, self.N) / ((pow(predator_distance, self.N)) * pow(self.h, self.N)) * (
                    math.pi / self.av) * (self.tv / self.t_min)
                if pd < random.random():
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
       
        # energy changes due to birth
        self.energy = self.energy - (self.max_energy / 2)
        child_params = mutate(deepcopy(self.evolvable_params))
        self.model.create_new_prey(child_params)
        



    def force_birth(self):
        n = 5
        summed_energy_neighbours = 0
        for agent in self.model.grid.get_neighbors((int(setup.GRID_WIDTH/2), int(setup.GRID_HEIGHT/2)),
                                                   include_center=True,
                                                   radius=setup.GRID_WIDTH+1):
            if agent.type == "prey":
                summed_energy_neighbours += agent.energy
        summed_energy_neighbours = max(summed_energy_neighbours, 1)
        prob_to_birth = math.pow(self.energy / summed_energy_neighbours, n)
        if np.random.random() > prob_to_birth:
            self.reproduce()

    def set_energy(self, new_energy):
        self.energy = new_energy

    def get_state(self):
        return self.state

    def is_alive(self):
        return self.state != Prey_State.DEAD

    def die(self):
        super().die()
        self.state = Prey_State.DEAD
        self.model.num_prey_agents -= 1
        self.model.prey
        return
