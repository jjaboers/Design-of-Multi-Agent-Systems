import mesa
from enum import Enum
import numpy as np
import random
import setup
from scipy.stats import truncnorm

class Prey_State(Enum):
    NOTHING = 1
    MOVING = 2
    FOODSCAN = 3
    MOVETOFOOD = 4
    EATING = 5
    SCANING = 6
    FLEEING = 7
    DEAD = 8


class PreyAgent(mesa.Agent):
    """An agent that is a prey, as described in the paper."""
    # non-evolvable parameters
    type = "prey"
    age = 0
    energy = 100000
    position = (random.randrange(setup.UI_WIDTH), random.randrange(setup.UI_HEIGHT))
    food_target = None
    zl = 50  # alignment zone
    dr = 0.9  # individual reach
    max_speed = 0.1
    max_neighbour_awareness = 50  # meters
    h = 5  # half-max distance detect predator
    N = 5  # scaling for predator detection
    em = 1  # metabolism
    min_energy = 0
    max_energy = 100000  # called eM in paper
    death_rate = 0.1
    max_age = 60 * 24 * 365 * 20  # 20 years expressed in minutes
    mutation_rate = 0.05
    is_safe = False
    waiting_time = 0  # TODO find initial value
    reaction_time = 1
    er = 2  # energy gained per food item TODO should this be in model?
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        print("INSIDE")
        print(getattr(self.model, 'num_prey_agents'))
        # evolvable parameters
        self.model = model
        self.state = Prey_State.NOTHING
        self.previous_state = Prey_State.NOTHING
        self.current_action_time_remaining = 0
        # descision making
        self.pv = 0  # predator scan, between 0 and 1, sd = 0.2
        self.pm = 0  # move after move, between 0 and 1, sd = 0.2
        self.pse = 0  # food scan after eat, between 0 and 1, sd = 0.2
        self.psn = 0  # food scan after no food, between 0 and 1, sd = 0.2
        self.pmtf = 0  # move to food, between 0 and 1, sd = 0.2
        # vigilance
        self.tv = 0  # scan duration, between 0.167 and 1.99, sd = 0.4
        self.av = 0  # scan angle, between 0 and 360, sd = 72
        # fleeing
        self.tp = 0  # flee duration, minimum 0, sd = 5
        # grouping
        self.zr = 20  # repulsion zone, between 0 and 50, sd = 10
        self.za = 30  # attractrion zone, between zr and 50, sd = 10
        self.aa = 72  # maximum turning angle for attraction, between 0 and 360, sd = 72
        self.ar = 72  # maximum turning angle for repulsion, between 0 and 360, sd = 72
        self.nr = 0  # tolerated neighbors, (0 or 1)
        # movement
        self.tm = 0  # move duration, between 0.167 and 1.99, sd = 0.4
        self.dm = 0  # move distance, minimum 0, sd = 3
        self.am = 0  # move angle, between 0 and 360, sd = 72
        # foraging
        self.df = 2  # search radius of forager
        self.af = 270  # search angle, angle between food and forward direction
        self.tf = 3  # foodscan duration

    # Function returns a random value for attraction/repulsion zone or angle of attraction/repulsion
    # Uses truncated normal distribution, takes range [lower, upper] and standard deviation (sd)
    def trunc_normal(self, lower, upper, sd):
        mu = upper - lower

        r = truncnorm.rvs(
            (lower - mu) / sd, (upper - mu) / sd, loc=mu, scale=sd, size=1)
        return r

    # SET EVOLVABLE PARAMS

    def set_repulsion_zone(self, repulsion):
        self.zr = repulsion

    def set_attraction_zone(self, attraction):
        self.za = attraction

    def set_repulsion_angle(self, repulsion):
        self.ar = repulsion

    def set_attraction_angle(self, attraction):
        self.aa = attraction

    # sets the initial values of evolvable parameters of the prey agent
    # TODO set all the parameters, currently only grouping is done
    def set_initial_evolvable_parameters(self):
        zr = self.trunc_normal(0, 50, 10)  # random value between 0 and 50, sd = 10
        self.set_repulsion_zone(zr)
        za = self.trunc_normal(zr, 50, 10)  # random value between self.zr and 50, sd = 10
        self.set_attraction_zone(za)
        aa = self.trunc_normal(0, 360, 72)  # random value between 0 and 360, sd = 72
        self.set_attraction_angle(aa)
        ar = self.trunc_normal(0, 360, 72)  # random value between 0 and 360, sd = 72
        self.set_repulsion_angle(ar)

    # STEP FUNCTION

    def step(self):
        self.age = self.age + 1
        self.energy = self.energy - self.em

        # Waiting time
        if self.is_safe == True:
            self.waiting_time = self.waiting_time - 1
        if self.waiting_time == 0:
            self.is_safe = False

        # Check if current action is over
        if self.current_action_time_remaining == 0:
            self.previous_state = self.state
            self.state = Prey_State.NOTHING
        else: # count down time remaining in action by 1
            self.current_action_time_remaining -= 1

        # if there is a predator or the neighbor flees, flee
        # takes precedence over everything, cutting short the current action
        # TODO Implement below if statement, not sure what condition is supposed to be yet
        # if something:
        #    self.state = Prey_State.FLEEING
        #    self.flee()

        # complete current action
        if self.state == Prey_State.DEAD:
            return
        elif self.state == Prey_State.MOVING:
            self.move()
        elif self.state == Prey_State.FOODSCAN:
            self.food_target = self.foodscan()
        elif self.state == Prey_State.MOVETOFOOD:
            self.move_to_food()
        elif self.state == Prey_State.EATING:
            self.energy += self.er
            self.food_target = None
        elif self.state == Prey_State.SCANING:
            self.scan()
        elif self.state == Prey_State.FLEEING and (self.is_safe is False):
            self.flee()
        # if current action complete or NONE, choose new action
        elif self.state == Prey_State.NOTHING:
            RAND = random.random()
            if RAND < self.pv or self.is_safe is True:
                self.state = Prey_State.SCANING
            else:
                if self.food_target is not None:
                    if self.distance(self.food_target) < self.dr:
                        self.state = Prey_State.EATING
                    else:
                        self.state = Prey_State.MOVETOFOOD
                else:
                    if self.previous_state == Prey_State.MOVING:
                        if RAND < self.pm:
                            self.state = Prey_State.MOVING
                        else:
                            self.state = Prey_State.FOODSCAN
                    elif self.previous_state == Prey_State.EATING:
                        if RAND < self.pse:
                            self.state = Prey_State.FOODSCAN
                        else:
                            self.state = Prey_State.MOVING
                    elif self.previous_state == Prey_State.MOVING:
                        if RAND < self.pse:
                            self.state = Prey_State.FOODSCAN
                        else:
                            self.state = Prey_State.MOVING

    # TODO choose random parent, force birth with no energy cost

    # PREY ACTIONS

    def move(self):
        # TODO add the grouping stuff to this, see sec 1.7.2 in sub paper)
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def distance(self, fooditem):
        return fooditem

    # calculates distance between self and food item

    # get new food target or not
    def foodscan(self):
        chosenitem = 100000

        # find all fooditems in range
        for fooditem in range(len(self.model.fooditems)):
            p = (self.tf * 60) / (
                np.pi * pow(self.df, 2) * (self.af / np.pi))  # we assume this function represents the vision
            RAND = random.random()
            if RAND < p:
                if self.distance(fooditem) < chosenitem:
                    chosenitem = fooditem

        return chosenitem

    def move_to_food(self, food_item):
        self.position = food_item.position - \
            (self.dr * abs(food_item.position - self.position))/2
        # TODO duration is distance moved * tM

    def eat(self, food_item):
        # resource items that are eaten disappear immediately (no half eating possible)
        self.model.remove_agents_food.append(food_item)
        # This should remove the agent from the grid, immediately to prevent it being eaten twice
        food_item.remove_agent()

    def scan(self):
        for predator in range(len(self.model.predators)):
            pass

    def flee(self):
        # No change in spatial position, safety is simply assumed
        self.is_safe = True
        self.current_action_time_remaining = self.reaction_time

    def reproduce(self):
        # Reproduction
        neighbours = self.model.grid.get_neighbors(self.position, include_center=True)
        n_children = int((len(neighbours)/2))
        self.model.create_prey(n_children)

        # if getattr(self.model, 'num_prey_agents') > 10 and self.energy >= self.max_energy:
        #     self.energy = self.energy - self.max_energy / 2
        #     a = PreyAgent(getattr(self.model, 'num_prey_agents') + 1, self.model)
        #     self.num_prey_agents = self.num_prey_agents + 1
        #     a.set_energy(self.max_energy / 2)
        #     # TODO offspring inherit all evolvable parameters + mutate, maybe make functions inherit() and evolve()
        # if self.model.num_prey_agents < 10:
        #     pass

    # SET NON-EVOLVABLE PARAMS

    def set_position(self, pos):
        self.position = pos

    def set_energy(self, new_energy):
        self.energy = new_energy
