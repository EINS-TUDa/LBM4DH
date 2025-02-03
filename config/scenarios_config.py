"""
This config file contains general settings for the scenarios,
    e.g. ambient temperature, supply and demand specifications, ...
    and the testcases used in the paper
"""

scen_config = {
    'Ta': 10.,                      # ambient temperature in °C
    'p_ret': 3.,                    # pressure at slack pp on return side
    'p_sup': 6.5,                   # pressure at slack pp on supply side
    'q_dem_mean': 200,              # mean demand for each consumer
    'dem_std': 0.2,                 # standard deviation for consumer demand
    'dem_correlation_factor': 5,    # demands are correlated based on phys. distance -> factor influences correlation
    'dem_min_temp': 55,             # minimal demand temperature
    'dem_nom_temp': 55,             # nominal demand temperature
    'dem_max_temp': 55,             # maximal demand temperature
    'sup_min_temp': 90,             # minimal supply temperature
    'sup_nom_temp': 110,            # nominal supply temperature
    'sup_max_temp': 130,            # maximal supply temperature
}

#%% testcases used in the paper:
test_cases = [
    {'scenario': 'ladder4', 'pp_pos': [0, 3], 'grid_dim': 300},
    {'scenario': 'ladder5', 'pp_pos': [0, 4], 'grid_dim': 300},
    {'scenario': 'ladder6', 'pp_pos': [0, 5], 'grid_dim': 300},
    {'scenario': 'ladder10', 'pp_pos': [0, 9], 'grid_dim': 300},
    {'scenario': 'ladder16', 'pp_pos': [0, 5, 10, 15], 'grid_dim': 300},
    {'scenario': 'cycle4', 'pp_pos': [0], 'grid_dim': 300},
    {'scenario': 'cycle5', 'pp_pos': [0], 'grid_dim': 300},
    {'scenario': 'cycle6', 'pp_pos': [0], 'grid_dim': 300},
    {'scenario': 'cycle10', 'pp_pos': [0], 'grid_dim': 300},
    {'scenario': 'cycle12', 'pp_pos': [0, 6], 'grid_dim': 300},
]
