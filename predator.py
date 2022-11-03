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
        # self.destination = None
        self.model = model
        self.nearby_predators = []
        self.nearby_prey = []
        self.evolve = evolve
        # set random initial direction
        self.direction = np.random.random(2)
        self.direction /= np.linalg.norm(self.direction)

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
        self.angle_move = params["angle_move"]
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
        print("-----------attack distance-------------", attack_distance)
        self.prey_detection_range = params["prey_detection_range"]
        self.attack_speed = params["attack_speed"]
        # predator specific parameters------------------------------------------

    def step(self):
        # only die and reproduced if predator evolves
        if self.evolve:
            too_old = self.age >= self.max_age
            starved = self.energy <= 0
            random_death = np.random.random() < self.death_rate
            if too_old or starved or random_death:
                self.die()
                return
            scanning = self.state == Predator_State.SCANNING
            searching = self.state == Predator_State.SEARCHING
            if scanning or searching:
                # returns false if not able to, skips a turn if reproduced
                if self.reproduce():
                    return

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

        predators_in_range = self.find_neighbors_in_range()
        self.group_move(predators_in_range)

    def chase(self):
        print("predator chasing")
        if self.target == None or not self.target.is_alive():
            self.set_state(Predator_State.SEARCHING)
            return

        if self.target.is_safe:
            self.set_state(Predator_State.SEARCHING)
            return

        target_pos = np.array(self.target.get_position())
        current_pos = np.array(self.get_position())
        dist = np.linalg.norm(target_pos - current_pos)

        if dist <= self.attack_distance:
            # not sure about waiting a step
            self.set_state(Predator_State.EATING)
            return

        self.direction = target_pos - current_pos
        self.direction = np.linalg.norm(self.direction)
        new_pos = self.max_speed * self.direction + self.position
        dist_travelled = np.linalg.norm(current_pos - new_pos)

        # Prevent overshooting
        if dist_travelled <= dist:
            new_pos = self.target.get_position()
        self.move((new_pos[0], new_pos[1]))

    def scan(self):
        if self.t_current_activity >= self.t_food_scan:
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
        print("predator eating")
        self.energy += self.target.get_energy()
        if self.energy < self.max_energy:
            self.energy = self.max_energy
        self.target.set_state(Prey_State.DEAD)
        self.target.die()
        self.target = None
        self.model.schedule.remove(self.target)
        self.model.grid.remove_agent(self.target)
        self.set_state(Predator_State.SEARCHING)

    def find_neighbors_in_range(self):
        predators = self.model.get_predators()
        if predators is None:
            return []
        predators_in_range = []
        for agent in predators:
            dist = np.linalg.norm(np.array(self.position)
                                  -
                                  np.array(agent.get_position())
                                  )
            if dist < self.max_neighbour_awareness:
                predators_in_range.append(agent)
        return predators_in_range

    # group move based on prey move
    def group_move(self, neighbors):
        # get number of actual neighbors within zones
        d_hat = np.array([0, 0])
        current_position = np.array(self.position)
        n_agents_rep = 0
        n_agents_attract = 0
        n_agents_align = 0
        sum_0 = np.array([0.0, 0.0])
        sum_1 = np.array([0.0, 0.0])
        sum_2 = np.array([0.0, 0.0])
        for agent in neighbors:
            agent_pos = np.array(agent.get_position())
            dist = np.linalg.norm(current_position -
                                  agent_pos
                                  )
            if self.r_repulsion >= dist:
                if dist != 0:
                    sum_0 += (agent_pos - current_position) / dist
                n_agents_rep += 1

            if self.alignment >= dist:
                if dist != 0:
                    sum_1 += (agent_pos - current_position) / dist
                n_agents_align += 1

            if self.r_attraction >= dist:
                sum_2 += np.array(agent.direction)
                n_agents_attract += 1
        abs_sum = math.sqrt((sum_0[0] * sum_0[0]) + (sum_0[1] * sum_0[1]))
        # if n_agents_repulsion_zone >= self.nr: TODO figure this out, what is self.nr???
        if abs_sum != 0:
            d_hat = - sum_0 / abs_sum
        else:
            d_hat = - sum_0
        # else
        sums = sum_1 + sum_2
        abs_sums = math.sqrt((sums[0] * sums[0]) + (sums[1] * sums[1]))
        if abs_sums != 0:
            d_hat = - sums / abs_sums
        else:
            d_hat = np.array([0, 0])

        dot_product = (self.direction[0] * d_hat[0]) + \
            (self.direction[1] * d_hat[1])
        v_abs = np.sqrt(
            (self.direction[0] * self.direction[0]) +
            (self.direction[1] * self.direction[1])
        )
        d_abs = np.sqrt((d_hat[0] * d_hat[0]) + (d_hat[1] * d_hat[1]))
        if v_abs == 0.0:
            angle = round(random.uniform(0, math.pi), 2)
        else:
            x = round(dot_product / v_abs * d_abs, 2)
            angle = math.acos(x)
            if angle < 0:
                angle = 360 - angle

        if angle <= self.angle_repulsion or angle <= self.angle_attraction:
            self.direction = d_hat
        else:
            vx = self.direction[0]
            vy = self.direction[1]
            # TODO it says "else turn aR or aA" but I'm not sure how to know which, so currently random?
            if random.random() < 0.5:
                a = self.angle_repulsion
            else:
                a = self.angle_attraction
            vx = vx * math.cos(a) - vy * math.sin(a)
            vy = vx * math.cos(a) + vy * math.sin(a)
            self.direction = [vx, vy]
        # random turn of a_M
        if random.random() < 0.5:
            t = - self.angle_move
        else:
            t = self.angle_move
        vx = self.direction[0]
        vy = self.direction[1]
        vx = vx * math.cos(t) - vy * math.sin(t)
        vy = vx * math.cos(t) + vy * math.sin(t)
        self.direction = np.array([vx, vy])

        # Get new position and make sure it is on the grid
        if (self.direction[0] + self.direction[1] != 0.0):
            # print("shapes: {0}, {1}, {2}".format(self.max_speed, self.direction, self.position))
            new_position = self.max_speed * self.direction + current_position
        else:
            # self.v_hat = np.array([self.pm, self.pm])
            self.direction = np.random.random(2)
            self.direction /= np.linalg.norm(self.direction)
            new_position = self.max_speed * self.direction + current_position
        new_position_rounded = new_position
        # Set new pos
        if (self.model.grid.out_of_bounds(new_position_rounded)):
            new_position_rounded = self.model.grid.torus_adj(
                new_position_rounded)
        self.move((new_position_rounded[0], new_position_rounded[1]))

    # Roaming in a group based on swarming
    def group_move_simple(self, predators_in_range):
        if len(predators_in_range) <= 0:
            return
        attraction_vec = self.attract_neighbors(predators_in_range)
        repulsion_vec = self.repulse_neighbors(predators_in_range)
        direction_vec = self.direction_neighbors(predators_in_range)
        self.direction += (
            attraction_vec +
            repulsion_vec +
            direction_vec
        )
        self.direction /= np.linalg.norm(self.direction)
        new_position = np.array(self.pos) + (self.direction * self.max_speed)
        self.move((new_position[0], new_position[1]))

    def random_move(self):
        # set random direction
        self.direction = np.random.random(2)
        self.direction /= np.linalg.nrom(self.direction)
        new_position = np.array(self.pos) + (self.direction * self.max_speed)
        self.move((new_position[0], new_position[1]))

    def move(self, new_position):
        # print("new_position: ", new_position)
        self.model.grid.move_agent(self, new_position)
        self.set_position(new_position)

    def attract_neighbors(self, neighbors):
        center = np.array([0.0, 0.0])
        for agent in neighbors:

            center += np.array(agent.pos)
        return center / len(center)

    def repulse_neighbors(self, neighbors):
        pos_vector = np.array(self.get_position())
        rep_vector = np.array([0.0, 0.0])
        for agent in neighbors:
            agent_pos = np.array(agent.get_position())
            dist = np.linalg.norm(pos_vector - agent_pos)
            if dist <= self.r_repulsion:
                rep_vector -= np.int64(agent_pos - pos_vector)
        return rep_vector

    def direction_neighbors(self, neighbors):
        direction_mean = np.array([0.0, 0.0])
        for agent in neighbors:
            direction_mean += np.int64(agent.direction)
        return direction_mean / len(neighbors)

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
            return False
        self.energy -= self.reproduction_cost
        params = predator_params.mutate_params(deepcopy(self.params))
        self.model.create_new_predator(params)
        return True
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
