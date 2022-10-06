import mesa
from mesa.time import RandomActivation
from predator import PredatorAgent
from prey import PreyAgent
from food import FoodAgent
from time import sleep


class Model(mesa.Model):
    """A model with some number of agents."""
    grid = None

    def __init__(self, N, width, height):
        super().__init__()  
        self.num_prey_agents = int(2*N/3)
        self.num_predator_agents = int(N/3)
        self.num_resources = width * height * 0.535  # factor found in paper

        self.grid = mesa.space.MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)

        self.create_prey(self.num_prey_agents)
        self.create_predators(self.num_predator_agents)
        self.create_food(self.num_resources)


    def step(self):
        """Advance the model by one step."""
        print(self.get_n_agents_per_type())
        self.schedule.step()  # model shuffles the order of the agents, then activates and executes each agent’s step method
        
        # TODO temp
        sleep(5)
        
        print("step")
        print(self.get_n_agents_per_type())
        

    def create_prey(self, num_prey_agents):
        # Create prey agents
        for i in range(num_prey_agents):
            print("create prey")

            a = PreyAgent(self.next_id(), self)
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
            a = PredatorAgent(self.next_id(), self)
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
            print("create food")
            a = FoodAgent(self.next_id(), self)
            self.schedule.add(a)
            cell = mesa.space.Grid.find_empty(self.grid)
            # print(cell)
            mesa.space.Grid.place_agent(self.grid, a, cell)
            a.set_position(cell)
            self.num_resources += 1

    def _get_num_prey_agents(self):
        return self.num_prey_agents

    def _set_num_prey_agents(self, num_prey_agents=None):
        self.num_prey_agents = num_prey_agents + self.num_prey_agents


    def get_n_agents_per_type(self):
        agent_buffer = self.schedule.agent_buffer()
        agent_counts = {"prey" : 0, "predator" : 0, "food" : 0}
        for agent in agent_buffer:
            agent_counts[agent.get_type()] += 1
        return agent_counts



