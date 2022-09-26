import mesa
from enum import Enum
import numpy as np
import random

# searching is roaming while scanning is looking
class Predator_State(Enum):
        SEARCHING   =   1
        SNEAKING    =   2
        CHASING     =   3
        SCANNING    =   4
        EATING      =   5
        DEAD        =   6

# this way all params can be manipulated in higher levels
default_params_predator = {
    "position"                  :   (0, 0)  ,
    "initial_energy"            :   100000  ,
    "search_radius"             :   10      ,
    "search_angle"              :   250     ,
    "t_food_scan"               :   3       ,
    "alignment"                 :   50      ,
    "reach"                     :   0.9     ,
    "max_speed "                :   0.12    ,
    "max_neighbour_awareness"   :   50      ,
    "energy_cost"               :   1       ,
    "max_energy"                :   100000  ,
    "death_rate"                :   0.1     ,
    "max_age"                   :   10512000,   # 60*24*365*20 = 20years in mins
    "mutation_rate"             :   0.05    ,
    "reproduction_requirement"  :   100000  ,   # max energy
    "reproduction_cost"         :   50000   ,   # half of max energy (paper)
    "offspring_energy"          :   50000   ,   # half of max energy (paper)
    "r_repulsion"               :   20      ,  
    "r_attraction"              :   30      ,
    "max_angle_attraction"      :   72      ,  
    "min_angle_attraction"      :   72      ,
    "attack_distance"           :   5       ,   # paper: 5, 7, 9
    "prey_detection_range"      :   50      ,   # not sure because angle and r 
    "attack_speed"              :   11.1    ,   # (m/s) prey paper, check wolf paper
}

# To ensure proportions are correct
def get_params_predator_scaled(scales = [1, 2, 2]):
    params = default_params_predator
    params["reproduction_requirement"] = params["max_energy"] / scales[0]
    params["reproduction_cost"] = params["max_energy"] / scales[1]
    params["offspring_energy"] = params["max_energy"] / scales[2]

class PredatorAgent(mesa.Agent):
    """An agent that is a predator"""

    def __init__(self, unique_id, model, params = default_params_predator):
        super().__init__(unique_id, model)
        # non-evolvable parameters
        # not variable parameters, these are always the same at construction 
        self.id = unique_id
        self.age = 0
        self.state = Predator_State.SEARCHING
        self.min_energy = 0
        # will be a prey object
        self.target = None
        # for moving to a location will be a tuple
        self.destination = None
        self.model = model
        self.speed_vector = np.array([0, 0])
        

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
        # perhaps make evolvable
        self.search_radius = params["search_radius"]  
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
        # the whole sneaking part might be redunant
        elif self.state == Predator_State.SNEAKING:
            self.sneak()
        elif self.state == Predator_State.CHASING:
            self.chase()
        elif self.state == Predator_State.SCANNING:
            self.scan()
        elif self.state == Predator_State.EATING:
            self.eat()
        
        self.age += 1

    def search(self):
        pass

    def sneak(self):
        pass

    def chase(self):
        pass
    
    def scan(self):
        pass

    def eat(self):
        pass

    def move(self):
        pass

    def set_target(self, pos):
        pass

    def set_state(self, state):
        pass

    def get_state(self):
        return self.state

    def is_alive(self):
        return self.state != Predator_State.DEAD
    
    # could be faster with euclidean on lists and custom dist function
    def is_at_destination(self):
        dist =  np.linalg.norm(
                    np.asarray(self.destination) - np.asarray(self.position))
        if dist < 0.01: # arbitrary threshold TODO check if works
            return True 
        return False
    
    # Also sets speed accordingly 
    def set_destination(self, destination):
        self.destination = destination
        vel = np.asarray(self.destination) - np.asarray(self.position)
        vel = vel / np.linalg.norm(vel) # nromalize vector
        self.speed_vector = vel * self.max_speed
    
    # TODO check bounds
    def set_random_destination(self):
        x = random.randint(0, self.model.grid.width - 1)
        y = random.randint(0, self.model.grid.height - 1)
        self.set_destination((x,y))

        
    # sexual or asexual ?
    def reproduce(self):
        pass

    def die(self): 
        self.state = Predator_State.DEAD
