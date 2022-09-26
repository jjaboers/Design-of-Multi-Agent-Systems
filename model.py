from mesa.time import RandomActivation
from predator import PredatorAgent
from prey import PreyAgent
from food import FoodAgent


class Model(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N, width, height):
        super().__init__()  # this fixes the problem
        self.num_prey_agents = N
        self.num_predator_agents = 2
        self.num_resources = width * height * 0.535  # probability found in paper
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.fooditems = []
        self.predators = []
        self.remove_agents_prey = []
        self.remove_agents_predator = []
        self.remove_agents_food = []

        # Place food items
        for resource in range(int(self.num_resources)):
            a = FoodAgent(resource, self)
            cell = mesa.space.Grid.find_empty(self.grid)
            print(cell)
            mesa.space.Grid.place_agent(self.grid, a, cell)
            a.set_position(cell)

        # Create prey agents
        for i in range(self.num_prey_agents):
            a = PreyAgent(i, self)
            self.schedule.add(a)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        # Create predator agents
        for i in range(self.num_predator_agents):
            a = PredatorAgent(i, self)
            self.schedule.add(a)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

    def step(self):
        """Advance the model by one step."""
        self.schedule.step()  # model shuffles the order of the agents, then activates and executes each agent’s step method
        # remove dead predators and prey, and eaten food
        for x in self.remove_agents_food:
            self.grid.remove_agent(x)
            self.schedule.remove(x)
            self.remove_agents_food.remove(x)
        for x in self.remove_agents_prey:
            self.grid.remove_agent(x)
            self.schedule.remove(x)
            self.remove_agents_prey.remove(x)
        for x in self.remove_agents_predator:
            self.grid.remove_agent(x)
            self.schedule.remove(x)
            self.remove_agents_predator.remove(x)
