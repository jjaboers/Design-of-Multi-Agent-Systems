import setup

# this way all params can be manipulated in higher levels
default_params_predator = {
    "position"                  :   (0, 0)  ,
    "initial_energy"            :   100000  ,
    "search_angle"              :   250     ,   # degrees TODO probably take out
    "t_food_scan"               :   3       ,
    "alignment"                 :   50      ,
    "reach"                     :   1.0     ,
    "max_speed"                 :   2       ,   # TODO find right val
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
    "search_duration"           :   3
}

def get_default_params_predator():
    return default_params_predator.copy()

# To ensure proportions are correct: Only execute once!
def scale_params_predator(scales = [1, 2, 2]):
    params = default_params_predator
    params["reproduction_requirement"] = params["max_energy"] / scales[0]
    params["reproduction_cost"] = params["max_energy"] / scales[1]
    params["offspring_energy"] = params["max_energy"] / scales[2]
    params["alignment"] *= setup.PROPORTION
    params["max_neighbour_awareness"] *= setup.PROPORTION
    params["r_repulsion"] *= setup.PROPORTION
    params["r_attraction"] *= setup.PROPORTION
    params["attack_distance"] *= setup.PROPORTION
    params["prey_detection_range"] *= setup.PROPORTION

