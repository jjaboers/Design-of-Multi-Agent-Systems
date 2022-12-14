import mesa
from mesa.time import RandomActivation
from predator import PredatorAgent
from prey import PreyAgent
from food import FoodAgent
from data_collector import DataCollector
from scipy.spatial import distance
import numpy as np
import random
import predator_params as pred_params
import uuid

class Model(mesa.Model):
    """A model with some number of agents."""
    # grid = None

    def __init__(self, N, width, height, attack_distance, evolve, n_prey = None, n_pred = None):
        super().__init__()
        # agent counts
        self.step_nr = 0
        if n_prey is None:
            self.num_prey_agents = int(2*N/3)
        else:
            self.num_prey_agents = n_prey
        
        if n_pred is None:
            self.num_predator_agents = int(N/3)
        else:
            self.num_predator_agents = n_pred

        self.num_resources = width * height * 0.535  # factor found in paper

        # init environment
        self.grid = mesa.space.ContinuousSpace(width, height, True)
        self.schedule = RandomActivation(self)

        # init agents
        self.attack_distance = attack_distance
        self.evolve = evolve
        self.create_prey(self.num_prey_agents)
        self.create_predators(self.num_predator_agents,
                              self.attack_distance, self.evolve)
        self.create_food(self.num_resources)

        # the schedule alredy has all agents, this might make every
        # timestep a little bit more efficient
        self.prey = []
        self.predators = []
        self.food = []

        # data
        self.data_collector = DataCollector(self)
        self.n_agents_per_type = None
        self.update_model_data()

        # kill list
        self.remove_agents_food = []

    def step(self):
        """Advance the model by one step."""
        self.data_collector.collect(self)

        # model shuffles the order of the agents, then activates and executes each agent’s step method
        self.schedule.step()
        self.update_model_data()
        for x in self.remove_agents_food:  # need to remove food agents taht have been eaten by prey
            self.schedule.remove(x)
            self.remove_agents_food.remove(x)
        self.step_nr += 1

    def create_prey(self, num_prey_agents):
        # Create prey agents

        for i in range(num_prey_agents):

            a = PreyAgent(self.next_id(), self)

            self.schedule.add(a)


            # Add the agent to a random grid cell
            x = self.random.random() * self.grid.x_max
            y = self.random.random() * self.grid.y_max
            pos = np.array((x, y))
            self.grid.place_agent(a, pos)
            a.set_position(pos)
            self.num_prey_agents += 1

    def create_new_prey(self, evolv_params):

        a = PreyAgent(self.next_id(), self, evolvable_params=evolv_params)

        a.set_energy(a.max_energy / 2)
        self.schedule.add(a)

        # Add the agent to a random grid cell
        x = self.random.random() * self.grid.x_max
        y = self.random.random() * self.grid.y_max
        pos = np.array((x, y))
        self.grid.place_agent(a, pos)
        a.set_position(pos)
        self.num_prey_agents += 1

    def create_new_predator(self, params):

        agent = PredatorAgent(self.next_id(), self,
                              self.attack_distance, params, evolve=self.evolve)

        agent.set_energy(agent.max_energy / 2)
        self.schedule.add(agent)
        x = random.uniform(0, self.grid.x_max)
        y = random.uniform(0, self.grid.y_max)
        self.grid.place_agent(agent, (x, y))
        agent.set_position((x, y))
        self.num_predator_agents += 1

    def create_predators(self, num_predator_agents, attack_distance, evolve):
        # Create predator agents
        params = pred_params.get_default_params_predator()

        for i in range(self.num_prey_agents+1, self.num_prey_agents + num_predator_agents):
            if evolve:
                params = pred_params.mutate_params(params)
            a = PredatorAgent(self.next_id(), self,
                              attack_distance, evolve=evolve)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.random() * self.grid.x_max
            y = self.random.random() * self.grid.y_max
            pos = np.array((x, y))
            self.grid.place_agent(a, pos)
            a.set_position(pos)
            self.num_predator_agents += 1

    def create_food(self, num_resources):
        # Place food items
        for resource in range(int(num_resources)):
            a = FoodAgent(self.next_id(), self)
            self.schedule.add(a)
            x = self.random.random() * self.grid.x_max
            y = self.random.random() * self.grid.y_max
            pos = np.array((x, y))
            self.grid.place_agent(a, pos)
            a.set_position(pos)
            self.num_resources += 1

    def get_n_agents_per_type(self):
        return self.n_agents_per_type

    def update_model_data(self):
        agent_buffer = self.schedule.agent_buffer()
        self.n_agents_per_type = {"prey": 0, "predator": 0, "food": 0}
        # loop twice is more efficient then reallocating in inner loop
        for agent in agent_buffer:
            self.n_agents_per_type[agent.get_type()] += 1
        # Allocate all lists (once for efficiency)
        self.prey = [None] * self.n_agents_per_type["prey"]
        self.predators = [None] * self.n_agents_per_type["predator"]
        self.food = [None] * self.n_agents_per_type["food"]
        #  update separate agent lists
        prey_idx, predator_idx, food_idx = 0, 0, 0
        agent_buffer = self.schedule.agent_buffer()
        for agent in agent_buffer:
            if agent.get_type() == "prey":
                self.prey[prey_idx] = agent
                prey_idx += 1
                continue
            if agent.get_type() == "predator":
                self.predators[predator_idx] = agent
                predator_idx += 1
                continue
            if agent.get_type() == "food":
                self.food[food_idx] = agent
                food_idx += 1

    def get_closest_agent_of_type_in_range(self, pos, type, range):
        agent_list = []
        if type == "prey":
            agent_list = self.prey
        if type == "predator":
            agent_list = self.predators
        if type == "food":
            agent_list = self.food

        ret_agent = None
        d_min = 50
        for agent in agent_list:
            dist = distance.euclidean(pos, agent.get_position())
            if dist <= d_min and dist <= range:
                ret_agent = agent
                d_min = dist
                break
        return ret_agent

    def get_predators(self):
        return self.predators

    def get_prey(self):
        return self.prey

    def get_global_overview(self):
        return self.data_collector.get_global_overview()
