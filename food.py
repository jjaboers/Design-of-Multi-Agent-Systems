from TypedAgent import TypedAgent
# TODO temp
import random

class FoodAgent(TypedAgent):
    """ Agent representing the resource items """
    type = "food"
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.position = (0, 0)
        self.energy_value = 2  # called Er in paper
        self.regrowth = None  

    def step(self):
        pass