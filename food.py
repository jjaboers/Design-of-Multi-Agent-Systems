class FoodAgent(mesa.Agent):
    """ Agent representing the resource items """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.position = (0, 0)
        self.energy_value = 2  # called Er in paper
        self.regrowth = None  # random timepoint in a year where they regrow

    def set_position(self, pos):
        self.position = pos
