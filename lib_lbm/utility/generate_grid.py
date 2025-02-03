import numpy as np
import regex as re
import json
from .DataClasses import GridModel

class node():
    def __init__(self, coordinates, index, name, nw_section):
        self.coordinates = np.round(np.array(coordinates), 3)
        self.index = index
        self.name = name
        self.nw_section = nw_section

    def get_string(self):
        return f'{{"type": "Node", "nw_section": "{self.nw_section}", "index": "{self.index}", ' \
               f'"name": "{self.name}", "coordinate": {[i for i in self.coordinates]}}}, \n'

class pipe():
    def __init__(self, index, from_node, to_node, iso_id):
        self.from_node = from_node
        self.to_node = to_node
        self.iso_id = iso_id
        self.length = np.round(np.linalg.norm(from_node.coordinates - to_node.coordinates), 3)
        self.index = index

    def get_string(self):
        return f'{{"type": "Edge", "edge control type": "passive", "index": "{self.index}", ' \
               f'"from": "{self.from_node.index}", "to": "{self.to_node.index}", "length [m]": {self.length}, ' \
               f'"nw_section": "{self.from_node.nw_section}", "Isoplus-ID": "{self.iso_id}",' \
               f'"pressure loss factor [-]": 1}}, \n'

class active_edge():
    def __init__(self, index, from_node, to_node):
        self.index = index
        self.from_node = from_node
        self.to_node = to_node

    def get_string(self):
        return f'{{"type": "Edge", "edge control type": "active", "index": "{self.index}", ' \
               f'"from": "{self.from_node.index}", "to": "{self.to_node.index}", "nw_section": "intern"}}, \n'

def generate_cycle_grid(n_sections, pp_pos, rad_grid_distance, iso_id):
    r_u = rad_grid_distance/2 /np.sin(np.pi/n_sections) # radius of the outer circle connecting all corners
    agn = []  # actual grid nodes
    agp = []  # actual grid pipes
    ae = []  # active edges
    aecp = []  # active edge connection pipes
    aecn = []  # active edge connection nodes

    # construct grid:
    for k in range(n_sections):
        x_pos_sup = r_u * np.cos(k/n_sections * 2 * np.pi)
        y_pos_sup = r_u * np.sin(k/n_sections * 2 * np.pi)
        x_pos_ret = x_pos_sup
        y_pos_ret = y_pos_sup - 2
        x_pos_sup_act = (r_u + 5) * np.cos(k/n_sections * 2 * np.pi)
        y_pos_sup_act = (r_u + 5) * np.sin(k/n_sections * 2 * np.pi)
        x_pos_ret_act = x_pos_sup_act
        y_pos_ret_act = y_pos_sup_act - 2

        # actual grid nodes:
        agn.append(node(coordinates=[x_pos_sup, y_pos_sup, 0], index=f'NS{k}', name=f'N_sup_{k}', nw_section='Sup'))
        agn.append(node(coordinates=[x_pos_ret, y_pos_ret, 0], index=f'NR{k}', name=f'N_ret_{k}', nw_section='Ret'))
        # actual grid pipes:
        if k >= 1:
            agp.append(pipe(index=f'S{k - 1}{k}', from_node=agn[-4], to_node=agn[-2], iso_id=iso_id))
            agp.append(pipe(index=f'R{k}{k-1}', from_node=agn[-1], to_node=agn[-3], iso_id=iso_id))
        # active edge:
        if k in pp_pos:
            aecn.append(
                node(coordinates=[x_pos_sup_act, y_pos_sup_act, 0], index=f'NS_PP{k}', name=f'N_sup_PP{k}', nw_section='Sup'))
            aecn.append(
                node(coordinates=[x_pos_ret_act, y_pos_ret_act , 0], index=f'NR_PP{k}', name=f'N_ret_PP{k}', nw_section='Ret'))
            ae.append(active_edge(index=f'PP{k}', from_node=aecn[-1], to_node=aecn[-2]))
            aecp.append(pipe(index=f'S_PP{k}', from_node=aecn[-2], to_node=agn[-2], iso_id=iso_id))
            aecp.append(pipe(index=f'R_PP{k}', from_node=agn[-1], to_node=aecn[-1], iso_id=iso_id))
        else:
            aecn.append(
                node(coordinates=[x_pos_sup_act, y_pos_sup_act, 0], index=f'NS_DEM{k}', name=f'N_sup_DEM{k}', nw_section='Sup'))
            aecn.append(
                node(coordinates=[x_pos_ret_act, y_pos_ret_act, 0], index=f'NR_DEM{k}', name=f'N_ret_DEM{k}', nw_section='Ret'))
            ae.append(active_edge(index=f'DEM{k}', from_node=aecn[-2], to_node=aecn[-1]))
            aecp.append(pipe(index=f'S_DEM{k}', from_node=agn[-2], to_node=aecn[-2], iso_id=iso_id))
            aecp.append(pipe(index=f'R_DEM{k}', from_node=aecn[-1], to_node=agn[-1], iso_id=iso_id))
    # close the loop:
    agp.append(pipe(index=f'S{k - 1}{0}', from_node=agn[-2], to_node=agn[0], iso_id=iso_id))
    agp.append(pipe(index=f'R{k - 2}{1}', from_node=agn[1], to_node=agn[-1], iso_id=iso_id))
    return [agn, agp, ae, aecp, aecn]


def generate_lader_grid(n_rungs, pp_pos, x_grid_distance, y_grid_distance, iso_id):
    agn = []   # actual grid nodes
    agp = []   # actual grid pipes
    ae = []    # active edges
    aecp = []  # active edge connection pipes
    aecn = []  # active edge connection nodes

    # construct grid:
    for k in range(n_rungs):
        x_pos = k*x_grid_distance
        y_pos_sup =  y_grid_distance/2
        y_pos_ret = -y_grid_distance/2
        # actual grid nodes:
        agn.append(node(coordinates=[x_pos, y_pos_sup, 0], index=f'NS{k}', name=f'N_sup_{k}', nw_section='Sup'))
        agn.append(node(coordinates=[x_pos, y_pos_ret, 0], index=f'NR{k}', name=f'N_ret_{k}', nw_section='Ret'))
        # actual grid pipes:
        if k >= 1:
            agp.append(pipe(index=f'S{k-1}{k}', from_node=agn[-4], to_node=agn[-2], iso_id=iso_id))
            agp.append(pipe(index=f'R{k}{k-1}', from_node=agn[-1], to_node=agn[-3], iso_id=iso_id))
        # active edge:
        if k in pp_pos:
            aecn.append(node(coordinates=[x_pos, y_pos_sup-1, 0], index=f'NS_PP{k}', name=f'N_sup_PP{k}', nw_section='Sup'))
            aecn.append(node(coordinates=[x_pos, y_pos_ret+1, 0], index=f'NR_PP{k}', name=f'N_ret_PP{k}', nw_section='Ret'))
            ae.append(active_edge(index=f'PP{k}', from_node=aecn[-1], to_node=aecn[-2]))
            aecp.append(pipe(index=f'S_PP{k}', from_node=aecn[-2], to_node=agn[-2], iso_id=iso_id))
            aecp.append(pipe(index=f'R_PP{k}', from_node=agn[-1], to_node=aecn[-1], iso_id=iso_id))
        else:
            aecn.append(node(coordinates=[x_pos, y_pos_sup - 1, 0], index=f'NS_DEM{k}', name=f'N_sup_DEM{k}', nw_section='Sup'))
            aecn.append(node(coordinates=[x_pos, y_pos_ret + 1, 0], index=f'NR_DEM{k}', name=f'N_ret_DEM{k}', nw_section='Ret'))
            ae.append(active_edge(index=f'DEM{k}', from_node=aecn[-2], to_node=aecn[-1]))
            aecp.append(pipe(index=f'S_DEM{k}', from_node=agn[-2], to_node=aecn[-2], iso_id=iso_id))
            aecp.append(pipe(index=f'R_DEM{k}', from_node=aecn[-1], to_node=agn[-1], iso_id=iso_id))

    return [agn, agp, ae, aecp, aecn]


def generate_star_grid(n_rays, ray_len, pp_pos, x_grid_distance, y_grid_distance, iso_id):
    agn = []   # actual grid nodes
    agp = []   # actual grid pipes
    ae = []    # active edges
    aecp = []  # active edge connection pipes
    aecn = []  # active edge connection nodes

    if type(ray_len) is int:
        ray_len = [ray_len for _ in range(n_rays)]

    shift_vec = np.array([x_grid_distance, 0])
    rotation_matrix = lambda pos: np.array([[np.cos(2*np.pi*pos/n_rays), -np.sin(2*np.pi*pos/n_rays)],
                                            [np.sin(2*np.pi*pos/n_rays),  np.cos(2*np.pi*pos/n_rays)]])

    def rename(input_string, name_change):
        name_splits = re.match(r"([a-z, _]+)([0-9]+)", input_string, re.I).groups()
        number = ''.join([str(int(c)+1) for c in name_splits[1]])
        return f'{name_splits[0]}{name_change}_{number}'

    def transform_node(node_obj, name_change):
        node_obj.coordinates[0:2] = rotation_matrix(ray) @ (node_obj.coordinates[0:2] + shift_vec)
        node_obj.index = rename(node_obj.index, name_change)
        node_obj.name = rename(node_obj.name, name_change)

    def transform_edge(edg_obj, name_change):
        edg_obj.index = rename(edg_obj.index, name_change)

    origin_sup = node(coordinates=[0, 0, 0], index=f'NS{0}', name=f'N_sup_{0}', nw_section='Sup')
    origin_ret = node(coordinates=[0, -y_grid_distance, 0], index=f'NR{0}', name=f'N_ret_{0}', nw_section='Ret')
    agn.append(origin_sup)
    agn.append(origin_ret)

    for ray in range(n_rays):
        pp_pos_ray = [pos[1] for pos in pp_pos if pos[0] == ray]
        n_rungs = ray_len[ray] - 1
        [agn_ray, agp_ray, ae_ray, aecp_ray, aecn_ray] = generate_lader_grid(n_rungs, pp_pos, x_grid_distance, y_grid_distance, iso_id)

        # rotate around origin and change names:
        for node_obj in agn_ray:
            transform_node(node_obj, str(ray))
        for node_obj in aecn_ray:
            transform_node(node_obj, str(ray))
        for edg_obj in agp_ray:
            transform_edge(edg_obj, str(ray))
        for edg_obj in aecp_ray:
            transform_edge(edg_obj, str(ray))
        for edg_obj in ae_ray:
            transform_edge(edg_obj, str(ray))

        # combine nodes with origin:
        agp.append(pipe(index=f'S{ray}_01', from_node=origin_sup, to_node=agn_ray[0], iso_id=iso_id))
        agp.append(pipe(index=f'R{ray}_10', from_node=agn_ray[1], to_node=origin_ret, iso_id=iso_id))

        # add 'new' ray to list of all nodes / edges
        agn.extend(agn_ray)
        agp.extend(agp_ray)
        ae.extend(ae_ray)
        aecp.extend(aecp_ray)
        aecn.extend(aecn_ray)

    return [agn, agp, ae, aecp, aecn]


def print_grid_to_file(file, agn, agp, ae, aecp, aecn):
    with open(file, 'w') as output_file:
        output_file.write('[{"comment": "actual grid nodes"}, \n')
        for n in agn:
            output_file.write(n.get_string())
        output_file.write('\n{"comment": "actual grid pipes"}, \n')
        for p in agp:
            output_file.write(p.get_string())
        output_file.write('\n{"comment": "active edges"}, \n')
        for e in ae:
            output_file.write(e.get_string())
        output_file.write('\n{"comment": "active edge connections"}, \n')
        for i, (e, n) in enumerate(zip(aecp, aecn)):
            output_file.write(n.get_string())
            o_str = e.get_string()
            if not i == len(aecp)-1:
                output_file.write(o_str)
            else:
                # very last entry: -> don't write the , at the end of the line (and the \n for what it is worth)
                output_file.write(o_str[:-3])

        output_file.write('\n ] \n')


def auto_set_iso_id(scenario, agp, aecp):
    """
        automatically adapts the size of pipes in the grid.

        this function solves the hydraulic model for 100 random demands;
        the pipe is chosen to be as small as possible but still be sufficient for every mass flow in the samples.
    """
    import pandas as pd
    import itertools
    import tensorflow as tf
    from .utility import load_scenario
    from lib_lbm.classic_solver import my_fixpoint_iteratror
    from lib_lbm.importance_sampling import setup_mf_sampling
    FI = my_fixpoint_iteratror()
    # solve_SE = SE_solver()

    grid, cycles, SE, d_dist, T_dist = load_scenario(scenario)
    mf_dist = setup_mf_sampling(d_dist, T_dist, SE, cycles)

    def get_mass_flow(mf_active, temperatures):
        while tf.reduce_sum(d_dist.signs * mf_active) < 0:
            mf_active = tf.squeeze(mf_dist.sample(1))
        SE.load_save_state()
        for dem in SE.demands.keys():
            ind = SE.dem_ind[dem]
            SE.set_active_edge_temperature(dem, temperatures[ind])
        # set temperature for heating - last entry in T_vector
        ind = [k for k in SE.heatings.keys()][0]
        SE.set_active_edge_temperature(ind, temperatures[-1])
        FI.solve_mf(SE, mf_active, cycles, loop=True)
        return SE.mf

    mf_samples = mf_dist.sample(100)
    T_samples = T_dist.sample(100)

    print('Calculating mass flows in the network. This might take several minutes, pls sand by ... ')

    mf = tf.map_fn(lambda args: get_mass_flow(args[0], args[1]), elems=[mf_samples, T_samples], dtype=tf.float64)
    mf = tf.squeeze(mf)
    pipes_db = pd.read_excel('grids/Library_Pipe_Parameters.xlsx', index_col=0, engine='openpyxl')
    pipe_cap = pipes_db.loc[:, 'maximaler Massenfluss'].iloc[2:]
    exp_mf_max = tf.reduce_max(tf.abs(mf), axis=0)
    # exp_mf_mean = np.abs(SE.mf.numpy())
    # exp_mf_max = 1.6 * exp_mf_mean # (exp. std ~ 0.2; 99% quantile at mu + 3 sigma)
    for edge in itertools.chain(agp, aecp):
        mf_max = exp_mf_max[SE.find_edge[edge.index]]       # maximum expected mass flow for this edge
        pos = np.argwhere(pipe_cap.values > mf_max)[0, 0]   # position of first pipe with capacity > mf_max
        edge.iso_id = pipe_cap.index[pos]                   # set isoplus-id accordingly


def loadGridModel(jsonPaths):
    """
    Loads district heating network data from json-file to  a MeFlexWärme GridModel
    @author: Friedrich
    """
    # load json-file. Note: Path must be given as raw-string: r"<Path>"
    nd = json.load(open(jsonPaths, "r"))
    nd = [el for el in nd if 'type' in el.keys()]  # Filter for elements that have the field 'type'
    nodes = [(nd[i]['index'], nd[i]) for i in range(len(nd)) if nd[i]['type'] == 'Node']
    edges = [(nd[i]['from'], nd[i]['to'], nd[i]) for i in range(len(nd)) if nd[i]['type'] == 'Edge']

    grid = GridModel()
    grid.add_nodes_from(nodes)
    grid.add_edges_from(edges)
    return grid


def __main__(name, kind, n_actives, pp_pos=[0], dim1_dist=50., dim2_dist=None, star_param={}, iso_id='auto'):
    """
        name: name of the save-file
        kind: either ladder or cycle
        n_dem: number of demands
        pp_pos: position of powerplants in the grid
        dim1_dist: distance between two demands
        dim2_dist: for ladder grids: distance between supply and return
        iso_id: if specified use this iso-id for all pipes. If 'auto' optimise diameter based on expected flow
    """
    if iso_id == 'auto':
        _iso_id = 'Isoplus_Std_DN40'
    else:
        _iso_id = iso_id

    if kind == 'ladder':
        [agn, agp, ae, aecp, aecn] = generate_lader_grid(n_rungs=n_actives, pp_pos=pp_pos, x_grid_distance=dim1_dist,
                                                         y_grid_distance=dim2_dist, iso_id=_iso_id)
    elif kind == 'cycle':
        [agn, agp, ae, aecp, aecn] = generate_cycle_grid(n_sections=n_actives, pp_pos=pp_pos, rad_grid_distance=dim1_dist,
                                                         iso_id=_iso_id)
    elif kind == 'star':
        [agn, agp, ae, aecp, aecn] = generate_star_grid(x_grid_distance=dim1_dist, y_grid_distance=dim2_dist,
                                                        iso_id=_iso_id, **star_param)
    else:
        raise Exception('unknown spcification for grid kind')

    if iso_id == 'auto':
        # print grid to file as this is the easiest way to convert it into a MeFlex-Grid-object
        print_grid_to_file(f'grids/{name}.json', agn, agp, ae, aecp, aecn)
        auto_set_iso_id(name, agp, aecp)
        print_grid_to_file(f'grids/{name}.json', agn, agp, ae, aecp, aecn)
    else:
        print_grid_to_file(f'grids/{name}.json', agn, agp, ae, aecp, aecn)

    # plot the resulting grid
    grid = loadGridModel(f'../../grids/{name}.json')
    grid.plot_topology_topview()

    from lib_lbm.utility.utility import load_scenario
    _, _, SE, _, _, _, _ = load_scenario(name)
    SE.report_state()


if __name__ == '__main__':
    specs = {
        'name': 'star20',
        'kind': 'star',
        'n_actives': 20,
        'pp_pos': [0, 5, 10, 15],
        'dim1_dist': 300,
        'dim2_dist': 10,
        # 'iso_id': 'auto',
        'iso_id': 'Isoplus_Std_DN40',
        'star_param': {
            'n_rays': 5,
            'ray_len': 3,
            'pp_pos': [(0,2), (3,3)]
        }
    }
    __main__(**specs)




