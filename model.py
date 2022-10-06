import mesa
from mesa.time import RandomActivation
from predator import PredatorAgent
from prey import PreyAgent
from food import FoodAgent


class Model(mesa.Model):
    """A model with some number of agents."""
    grid = None

    def __init__(self, N, width, height):
        super().__init__()  # this fixes the problem

        self.num_prey_agents = int(2*N/3)
        self.num_predator_agents = int(N/3)
        self.num_resources = width * height * 0.535  # probability found in paper
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.fooditems = []
        self.predators = []
        self.prey_list = []
        self.remove_agents_prey = []
        self.remove_agents_predator = []
        self.remove_agents_food = []

        self.create_prey(self.num_prey_agents)
        self.create_predators(self.num_predator_agents)
        self.create_food(self.num_resources)


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

    def create_prey(self, num_prey_agents):
        # Create prey agents
        for i in range(num_prey_agents):
            print("create prey")
            a = PreyAgent(i, self)
            self.schedule.add(a)

            # Add the agent to a random grid cell
            cell = mesa.space.Grid.find_empty(self.grid)
            # self.grid.place_agent(a, (x, y))
            mesa.space.Grid.place_agent(self.grid, a, cell)
            a.set_position(cell)
            self.num_prey_agents += 1

    def create_predators(self, num_predator_agents):
        # Create predator agents
        for i in range(self.num_prey_agents+1, self.num_prey_agents + num_predator_agents):
            print("create predator")
            a = PredatorAgent(i, self)
            self.schedule.add(a)

            # Add the agent to a random grid cell
            # x = self.random.randrange(self.grid.width)
            # y = self.random.randrange(self.grid.height)
            cell = mesa.space.Grid.find_empty(self.grid)
            # self.grid.place_agent(a, (x, y))
            mesa.space.Grid.place_agent(self.grid, a, cell)
            a.set_position(cell)
            self.num_predator_agents += 1

    def create_food(self, num_resources):
        # Place food items
        for resource in range(int(num_resources)):
            a = FoodAgent(resource, self)
            cell = mesa.space.Grid.find_empty(self.grid)
            # print(cell)
            mesa.space.Grid.place_agent(self.grid, a, cell)
            a.set_position(cell)
            self.num_resources += 1

    def _get_num_prey_agents(self):
        return self.num_prey_agents

    def _set_num_prey_agents(self, num_prey_agents=None):
        self.num_prey_agents = num_prey_agents + self.num_prey_agents

