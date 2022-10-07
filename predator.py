import mesa
from enum import Enum
import numpy as np
from TypedAgent import TypedAgent
from prey import Prey_State
import predator_params
from scipy.spatial.distance import euclidean as dist

# searching is roaming while scanning is looking
class Predator_State(Enum):
        SEARCHING   =   1
        CHASING     =   2
        SCANNING    =   3
        EATING      =   4
        DEAD        =   5


class PredatorAgent(TypedAgent):
    """An agent that is a predator"""
    def __init__(self, unique_id, model, 
                    params = predator_params.default_params_predator):
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
        self.destination = None
        self.model = model
        self.nearby_predators = []
        self.nearby_prey = []
        

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
        self.max_angle_attraction = params["max_angle_attraction"]  
        self.min_angle_attraction = params["min_angle_attraction"]
        # evolvable parameters--------------------------------------------------

        # predator specific parameters------------------------------------------
        self.attack_distance = params["attack_distance"]
        self.prey_detection_range = params["prey_detection_range"]
        self.attack_speed = params["attack_speed"]
        # predator specific parameters------------------------------------------
    
    # TODO check duration of step as age is in minutes
    def step(self):
        if self.age >= self.max_age:
            self.die()
        # TODO implement death_rate 
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
        
        possible_steps = self.model.grid.get_neighborhood(
            self.position,
            moore=True,
            include_center=False)
        new_position = self.random.choice(possible_steps)
        self.move(new_position)
        
    # with current fleeing system more like charge
    def chase(self):
        if self.target == None or not self.target.is_alive():
            self.set_state(Predator_State.SEARCHING)
            return 
        if self.target.is_safe:
            self.set_state(Predator_State.SEARCHING)
            return 
        if dist(self.position, self.target.get_position()) <= self.reach:
            self.set_state(Predator_State.EATING)
            return 

        for step in range(self.max_speed):
            possible_steps = self.model.grid.get_neighborhood(
                self.position,
                moore=True,
                include_center=True)
            # Select position closest to target position
            new_position = possible_steps[
                                        np.argmin(
                                        [
                                        dist(self.target.get_position(), pos) 
                                        for pos in possible_steps
                                        ]
                                        )
                                        ]
            self.move(new_position)


    
    def scan(self):
        if self.t_current_activity >= self.search_duration:
            self.set_state(Predator_State.SEARCHING)
            return 
        agent = self.model.get_closest_agent_of_type_in_range(  
                                            pos = self.position , 
                                            type = "prey"       , 
                                            range = self.prey_detection_range
                                            )
        if agent != None:
            self.target = agent
            self.set_state(Predator_State.CHASING)


    def eat(self):
        self.energy += self.target.get_energy()
        if self.energy < self.max_energy:
            self.energy = self.max_energy 
        self.target.set_state(Prey_State.DEAD)
        self.model.schedule.remove(self.target)
        self.set_state(Predator_State.SEARCHING)

    
    # TODO implement repulsion etc
    def move(self, new_position):
        self.model.grid.move_agent(self, new_position)
        self.set_position(new_position)

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
        pass
    
    def die(self): 
        super().die()
        self.set_state(Predator_State.DEAD)
