import mesa
import numpy as np
import random
import setup

class PreyAgent(mesa.Agent):
    """An agent that is a prey, as described in the paper."""
    # non-evolvable parameters
    age = 0
    energy = 100000  # TODO what energy level do they start with?
    position = (random.randrange(setup.UI_WIDTH), random.randrange(setup.UI_HEIGHT))
    df = 2  # search radius of forager, TODO find initial value
    af = 270  # search angle, angle between food and forward direction
    tf = 3  # foodscan duration, TODO find initial value
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
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        print("INSIDE")
        print(getattr(self.model, 'num_prey_agents'))
        # evolvable parameters
        self.model = model
        self.zr = 20  # repulsion zone, affected by evolution, TODO set it to random value between 0 and 50, sd = 10
        self.za = 30  # attractrion zone, affected by evoluiton, TODO set it to random value between self.zr and 50, sd = 10
        self.aa = 72  # maximum turning angle for attraction TODO set between 0 and 360, sd = 72
        self.ar = 72  # maximum turning angle for repulsion TODO set between 0 and 360, sd = 72

    def step(self):
        self.age = self.age + 1
        self.energy = self.energy - self.em

        # Waiting time
        if self.is_safe == True:
            self.waiting_time = self.waiting_time - 1
        if self.waiting_time == 0:
            self.is_safe = False


    # TODO choose random parent, force birth with no energy cost

    def move(self):
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
        pass
        # No change in spatial position, safety is simply assumed
        # TODO duration = reactiontime (1 second)
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

    def set_energy(self, new_energy):
        self.energy = new_energy
