import mesa

class TypedAgent(mesa.Agent):
	"""An agent that is a predator"""
	position = None
	def __init__(self, unique_id, model):
		super().__init__(unique_id, model)


	def set_position(self, pos):
		self.position = pos


	def get_type(self):
		return self.type