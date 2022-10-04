import mesa


class PredatorAgent(mesa.Agent):
    """An agent that is a predator"""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # non-evolvable parameters
        # self.position(0, 0)
        # evolvable parameters
        # TODO Predator incomplete
