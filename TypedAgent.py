import mesa
from copy import deepcopy
# TODO
# 	- add energy
# 	- add all common params
# 	- 
class TypedAgent(mesa.Agent):
	"""An agent that is a predator"""
	position = None
	params = []
	age = 0
	energy = 0
	def __init__(self, unique_id, model, params = None):
		super().__init__(unique_id, model)
		if params is not None:
			self.params = deepcopy(params)
		else:
			self.params = None

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
	
	def get_energy(self):
		return self.energy

	def get_position(self):
		return self.position
	
	def set_energy(self, energy):
		self.energy = energy