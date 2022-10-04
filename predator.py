import mesa


class PredatorAgent(mesa.Agent):
    """An agent that is a predator"""
    type = "predator"
    position = (0,0)

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # non-evolvable parameters
        # self.position(0, 0)
        # evolvable parameters
        # TODO Predator incomplete

    def set_position(self, pos):
        self.position = pos
