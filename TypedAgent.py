import mesa

# TODO
# 	- add energy
# 	- add all common params
# 	- 
class TypedAgent(mesa.Agent):
	"""An agent that is a predator"""
	position = None
	params = []
	age = 0
	def __init__(self, unique_id, model):
		super().__init__(unique_id, model)


	def set_position(self, pos):
		self.position = pos

	def get_type(self):
		return self.type

	def get_params(self):
		return self.params

	def die(self):
		self.model.schedule.remove(self)

	def step(self):
		self.age += 1