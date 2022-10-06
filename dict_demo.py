from random import randint
import predator

params = predator.get_default_params_predator()

params["r_repulsion"] = 40

id = 0
model = None

pred = predator.PredatorAgent(unique_id=id, model=model, params=params)

print(params["r_repulsion"], predator.default_params_predator["r_repulsion"])