"""

This file contains the load routines for the scenarios.
Only the function 'load_scenario' is public, all other functions are private and should not be used outside this file.

"""

import numpy as np
import json
import pandas as pd
import networkx as nx
import re
import tensorflow as tf
import itertools
from lib_lbm.utility.DataClasses import GridModel
from lib_lbm.classic_solver import SE_solver
from lib_lbm.utility.state_equations import StateEquations
from lib_lbm.utility.DataClasses import ZeroTruncatedMultivariateNormal, UniformDistribution
from config.scenarios_config import scen_config


# lib for pipe parameter:
pipes_db = pd.read_excel('grids/Library_Pipe_Parameters.xlsx', index_col=0, engine='openpyxl')

def _loadGridModel(jsonPaths):
    """
    Loads district heating network data from json-file to a GridModel
    @author: Friedrich
    """
    nd = json.load(open(jsonPaths, "r"))
    nd = [el for el in nd if 'type' in el.keys()]  # Filter for elements that have the field 'type'
    nodes = [(nd[i]['index'], nd[i]) for i in range(len(nd)) if nd[i]['type'] == 'Node']
    edges = [(nd[i]['from'], nd[i]['to'], nd[i]) for i in range(len(nd)) if nd[i]['type'] == 'Edge']

    grid = GridModel()
    grid.add_nodes_from(nodes)
    grid.add_edges_from(edges)
    return grid


def _find_all_cycles(graph_input):
    '''
    Returns a List with one entry for each passive loop in the grid:
        each entry is a list containing all edges that form the loop
    '''
    g = nx.DiGraph(graph_input)
    cycles = []
    while True:
        try:
            cycle = nx.algorithms.cycles.find_cycle(g, orientation='ignore')
            cycles.append(cycle)
        except nx.exception.NetworkXNoCycle:  # exception appears if no cycle is found
            break
        for u, v, direction in cycle:
            g.remove_edge(u, v)
    return cycles

class ScenarioNotDefinedException(Exception):
    def __init__(self, scenario_name):
        # Call the base class constructor with the parameters it needs
        message = f'The Scenario {scenario_name} is not defined'
        super().__init__(message)


def _scenario_configuration(scenario_name):
    '''
    This function parses the grid file and converts the specifies applies defined at the beginning of this file
    to actual distributions, supply/demand sets, ...
    '''

    n = int(re.findall('\d+', scenario_name)[-1])  # regex finds all numbers in the string
    if 'ladder' in scenario_name:
        grid_type = 'ladder'
        grid_file = f'grids/ladder{n}.json'
    elif 'cycle' in scenario_name:
        grid_type = 'cycle'
        grid_file = f'grids/cycle{n}.json'

    else:
        raise Exception('cant parse scenario specification')

    pp_ind = dict()
    dem_ind = dict()
    active_edge = dict()
    dem_ind_list = []
    try:
        with open(grid_file, 'r') as gf:
            for l in gf:
                if 'edge control type": "active' in l:
                    if 'PP' in l:
                        ind = re.findall('PP\w+', l)[0]
                        pp_ind[ind] = int(ind[-1])
                    if 'DEM' in l:
                        ind = re.findall('DEM\w+', l)[0]
                        dem_ind[ind] = int(ind[-1])
                        dem_ind_list.append(int(ind[-1]))
                    # active_edge counts the position of each demand in the covariance matrix
                    if len(pp_ind) == 0:
                        active_edge[ind] = int(ind[-1])
                    else:
                        active_edge[ind] = int(ind[-1])-1
    except FileNotFoundError:
        raise ScenarioNotDefinedException(f'{grid_type}{n}')

    h_key = list(pp_ind.keys())[0]
    active_edge = {key: active_edge[key] for key in active_edge.keys() if not key == h_key}

    q_dem = scen_config['q_dem_mean']
    q_pp = -q_dem * len(dem_ind)/len(pp_ind)
    d_prior_mean = tf.constant([q_dem if ind in dem_ind.keys() else q_pp for ind in active_edge.keys()], dtype=tf.float64)

    d_ret_temps_d = {k: {'min': scen_config['dem_min_temp'], 'nom': scen_config['dem_nom_temp'], 'max': scen_config['dem_max_temp']} for k in dem_ind.keys()}
    d_ret_temps_h = {k: {'min': scen_config['sup_min_temp'], 'nom': scen_config['sup_nom_temp'], 'max': scen_config['sup_max_temp']} for k in pp_ind.keys()}
    d_ret_temps = {**d_ret_temps_d, **d_ret_temps_h}

    # distance between demands:
    dist = np.array([[abs(i-j) for i in dem_ind.values()] for j in dem_ind.values()])
    if grid_type=='cycle':
        # cyclic distance -> go in both directions and take smaller distance
        dist = np.minimum(dist, n - dist)
    cor = np.exp(-scen_config['dem_correlation_factor'] * dist / np.max(dist))

    cor_np = np.eye(len(active_edge))
    for (i, row), (j, col) in itertools.product(enumerate(dem_ind.keys()), (enumerate(dem_ind.keys()))):
        cor_np[active_edge[row], active_edge[col]] = cor[i, j]
    cor = tf.constant(cor_np)

    stds = scen_config['dem_std'] * d_prior_mean
    d_prior_cov = tf.transpose(stds * cor) * stds

    heatings = {h_key: {'Power': -q_dem*len(dem_ind)/len(pp_ind),  'Temperature': d_ret_temps[h_key]['nom'],
                        'T_min': d_ret_temps[h_key]['min'], 'T_max': d_ret_temps[h_key]['max']}}
    demands = {key: {'Power': q_dem, 'Temperature': d_ret_temps[key]['nom'],
                     'T_min': d_ret_temps[key]['min'], 'T_max': d_ret_temps[key]['max']} for key in dem_ind.keys()}
    demands = {**demands, **{key: {'Power': -q_dem*len(dem_ind)/len(pp_ind), 'Temperature': d_ret_temps[key]['nom'],
                                   'T_min': d_ret_temps[key]['min'], 'T_max': d_ret_temps[key]['max']} for key in pp_ind.keys()
                             if key != h_key}}

    fix_dp = {f'NS_{list(pp_ind.keys())[0]}': scen_config['p_sup'], f'NR_{list(pp_ind.keys())[0]}': scen_config['p_ret']}
    Ta = tf.constant(scen_config['Ta'], dtype=tf.float64)

    return grid_file, demands, heatings, fix_dp, Ta, d_prior_mean, d_prior_cov


def load_scenario(scenario_name, verbose=False):
    """
    this function sets up a State Equation Object and the prior distributions
        :param scenario_name:  specification, which scenario should be loaded
        :return: grid: Grid object,
                 cycles: list of edges forming a passive loop,
                 SE: State-Equations object,
                 d_prior_dist: prior demand distribution,
                 T_prior_dist: prior temperature distribution
    """

    # load grid specific parameter, i.e. grid layout, pp position ...
    grid_file, demands, heatings, fix_dp, Ta, d_prior_mean, d_prior_cov = _scenario_configuration(scenario_name)
    grid = _loadGridModel(grid_file)

    # identify all cycles in the passive parts of the grid (looping pipes)
    sup_graph = nx.DiGraph(grid.subgraph((node for node, data in grid.nodes(data=True) if data['nw_section'] == 'Sup')))
    ret_graph = nx.DiGraph(grid.subgraph((node for node, data in grid.nodes(data=True) if data['nw_section'] == 'Ret')))
    cycles = []
    cycles.extend(_find_all_cycles(sup_graph))
    cycles.extend(_find_all_cycles(ret_graph))
    if not cycles == []:
        cycles = [[(grid.edges[(e[0], e[1])]['index'], e[2]) for e in c] for c in cycles]

    # store nodes and edges in list format for easier access in the SE object:
    nodes = []
    edges = []

    for node_idx in grid.nodes():
        nodes.append(grid.nodes[node_idx])

    for edge_idx in grid.edges():
        edge_vals = grid.edges[edge_idx]
        if edge_vals['edge control type'] == 'passive':
            edge_vals['temp_loss_coeff'] = pipes_db.loc[edge_vals['Isoplus-ID'], 'Norm-Wärme-übergangskoeffizent']
            edge_vals['diameter'] = pipes_db.loc[edge_vals['Isoplus-ID'], 'hydraulischer Durchmesser']
            if edge_vals['nw_section'] == 'sup':
                edge_vals['fd_nom'] = pipes_db.loc[edge_vals['Isoplus-ID'], 'fd_nom 110C']
            else:
                edge_vals['fd_nom'] = pipes_db.loc[edge_vals['Isoplus-ID'], 'fd_nom 65C']
            edge_vals['bend_factor'] = edge_vals.get('pressure loss factor[-]', 1)
        # edges contains all active and passive edges
        edges.append(edge_vals)

    # construct state equations object
    SE = StateEquations(edges=edges, nodes=nodes, demands=demands, heatings=heatings, fix_dp=fix_dp, Ta=Ta)
    SE.set_init_state()

    # solve for mean conditions and save resulting state
    solve_SE = SE_solver()
    solve_SE(SE, cycles, verbose=verbose)
    SE.set_save_state()

    # setup distributions for heat powers and feed in temperatures:
    d_prior_dist = ZeroTruncatedMultivariateNormal(loc=d_prior_mean, scale_tril=tf.linalg.cholesky(d_prior_cov),
                                                   validate_args=True, name='prior_demand_distribution')

    T_min, T_max = np.zeros((len(demands) + 1)), np.zeros((len(demands) + 1))
    for dem in demands.keys():
        T_min[SE.dem_ind[dem]] = demands[dem]['T_min']
        T_max[SE.dem_ind[dem]] = demands[dem]['T_max']
    T_min[-1] = [heatings[k]['T_min'] for k in heatings.keys()][0]
    T_max[-1] = [heatings[k]['T_max'] for k in heatings.keys()][0]
    T_prior_dist = UniformDistribution(low=tf.cast(T_min, dtype=tf.float64), high=tf.cast(T_max, dtype=tf.float64))

    return grid, cycles, SE, d_prior_dist, T_prior_dist