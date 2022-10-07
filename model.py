import mesa
from mesa.time import RandomActivation
from predator import PredatorAgent
from prey import PreyAgent
from food import FoodAgent
from data_collector import DataCollector
from scipy.spatial import distance

class Model(mesa.Model):
    """A model with some number of agents."""
    grid = None

    def __init__(self, N, width, height):
        super().__init__()  
        # agent counts
        self.num_prey_agents = int(2*N/3)
        self.num_predator_agents = int(N/3)
        self.num_resources = width * height * 0.535  # factor found in paper
        
        # init environment
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)

        # init agents
        self.create_prey(self.num_prey_agents)
        self.create_predators(self.num_predator_agents)
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
        


    def step(self):
        """Advance the model by one step."""
        self.data_collector.collect(self)
        self.schedule.step()  # model shuffles the order of the agents, then activates and executes each agent’s step method
        self.update_model_data()
        # print(self.predators)
        # print(self.prey)
        # print(self.food)
        print(self.n_agents_per_type)
        

    def create_prey(self, num_prey_agents):
        # Create prey agents
        for i in range(num_prey_agents):
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
            a = PredatorAgent(self.next_id(), self)
            self.schedule.add(a)
            cell = mesa.space.Grid.find_empty(self.grid)
            mesa.space.Grid.place_agent(self.grid, a, cell)
            a.set_position(cell)
            self.num_predator_agents += 1

    def create_food(self, num_resources):
        # Place food items
        for resource in range(int(num_resources)):
            a = FoodAgent(self.next_id(), self)
            self.schedule.add(a)
            cell = mesa.space.Grid.find_empty(self.grid)
            mesa.space.Grid.place_agent(self.grid, a, cell)
            a.set_position(cell)
            self.num_resources += 1


    def get_n_agents_per_type(self):
        return self.n_agents_per_type

    def update_model_data(self):
        agent_buffer = self.schedule.agent_buffer()
        self.n_agents_per_type = {"prey" : 0, "predator" : 0, "food" : 0}
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
        d_min = 10000000
        for agent in agent_list:
            dist = distance.euclidean(pos, agent.get_position())
            if dist <= d_min and dist <= range:
                ret_agent = agent 
                d_min - dist 
        return ret_agent
            
            

                


        
        




