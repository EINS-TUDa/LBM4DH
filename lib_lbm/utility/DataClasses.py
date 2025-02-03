'''
This file contains basic data classes

GridModel: Data Class describing the Grid
DNNData: Stores training data for DNN training
UniformDistribution: Uniform distribution to model temperature priors, copy of tfd.Uniform with overwritten __repr__
ZeroTruncatedMultivariateNormal: Zero truncated Normal distribution used to model heat power priors
'''


import networkx as nx
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import lib_lbm.se_NN_lib as NN
import matplotlib.pyplot as plt
from typing import List, Dict
import warnings
tfd = tfp.distributions

class GridModel(nx.DiGraph):
    """
    Stores all data about the pipes in a network

    Uses modul "networkx.Graph" as base. Read networkx docs for further information.

    Author: Friedrich P. Bott A.
    """
    def __init__(self, networkData=None, filepath=None, name="grid"):
        super().__init__()
        if filepath:
            raise NotImplementedError("Cannot yet process filepaths.")
        if networkData:
            networkData = [el for el in networkData if
                           'type' in el.keys()]  # Filter for elements that have the field 'type'
            nodes = [(networkData[i]['index'], networkData[i]) for i in range(len(networkData)) if
                     networkData[i]['type'] == 'Node']
            edges = [(networkData[i]['from'], networkData[i]['to'], networkData[i]) for i in range(len(networkData)) if
                     networkData[i]['type'] == 'Edge']
            if not len(nodes) + len(edges) == len(networkData):
                raise Exception('Some network elements are neither type "node" nor "edge"')

            self.add_nodes_from(nodes)
            self.add_edges_from(edges)
            self.json = networkData
            self.name = name

    def get_passive_grid(self):
        """
        returns a grid object that contains only the passive grid elements
        """
        if not hasattr(self, '_passive_grid'):
            sup_nodes = (node for node, data in self.nodes(data=True) if data['nw_section'] == 'Sup')
            ret_nodes = (node for node, data in self.nodes(data=True) if data['nw_section'] == 'Ret')
            self._sup_graph = self.subgraph(sup_nodes)
            self._ret_graph = self.subgraph(ret_nodes)
            self._passive_grid = nx.compose(self._sup_graph, self._ret_graph)
        return self._passive_grid

    def get_section(self, section):
        """Returns supply or return section of network.

        :param section: Either 'Sup' or 'Ret'
        :type section: str
        :return: Grid model of sub section
        :rtype: GridModel
        """
        nodes = {n for n in self.nodes if
                 self.nodes[n]['nw_section'] == section}  # select supply section only for grid constraints
        return self.subgraph(nodes)

    def get_active_nodes(self):
        "returns nodes that active edges are connected to"
        active_nodes = []
        for u, v, ctrl_type in self.edges(data='edge control type'):
            if ctrl_type == 'active':
                active_nodes.extend([u, v])

        return active_nodes

    def find_all_cycle(self):
        """
        returns a list containing all cycles within the grid, ignoring active edges
        """
        search_grid = self.get_passive_grid()
        cycles = []
        while True:
            try:
                cycle = nx.algorithms.cycles.find_cycle(search_grid, orientation='ignore')
                cycles.append(cycle)
            except nx.exception.NetworkXNoCycle:  # exception appears if no graph is found
                break
            for u, v, direction in cycle:
                search_grid.remove_edge(u, v)
        return cycles

    def all_edges(self, node) -> List:
        """Returns all edges connected to given nodes,
        regardless of starting or ending at that nodes
        (in contrast to self.edges(node)).
        Returns edges in correct direction!"""
        edges = list(self.in_edges(node))
        edges.extend(list(self.out_edges(node)))
        return edges

    def has_edge_both_dir(self, u, v):
        if self.has_edge(u, v):
            return True
        elif self.has_edge(v, u):
            return True
        else:
            return False

    def has_edge_by_index(self, idx):
        for u, v in self.edges:
            if self.edges[u, v]['index'] == idx:
                return True
        return False

    def edge_by_index(self, idx):
        for u, v in self.edges:
            if self.edges[u, v]['index'] == idx:
                return self.edges[u, v]
        # if not found:
        raise KeyError("edge with index %s not found" % idx)

    def node_edge_idx(self, node, mode):
        """Returns all indeces of edges that are connected to a node. Lets you chose which edges you need by parameter "mode":
        Going to, starting at or all edges of node.

        :param node: id of node
        :type node: str
        :param mode: Either "to", "from" or "all"
        :return: edge indeces
        :rtype: list
        """
        if mode == 'to':
            edges = self.in_edges(node)
        elif mode == 'from':
            edges = self.out_edges(node)
        elif mode == 'all':
            edges = set(self.in_edges(node))
            edges.update(set(self.out_edges(node)))
        else:
            raise ValueError("Unknown mode \"%s\"" % mode)
        idx = [self.edges[e]['index'] for e in edges]

        return idx

    def edge_idx_by_ctrl_type(self, ctrl_type):
        """Returns list of indeces of all edges with given control type.

        :param ctrl_type: 'active' or 'passive' (not pT, QT etc.)
        :type ctrl_type: str
        :return: list of indeces
        :rtype: List[str]
        """
        return [e['index'] for e in self.edges.values() if e['edge control type'] == ctrl_type]

    def plot_topology_topview(self, coloring: Dict = {}):
        """
            coloring: dict {color : list[node index]}
            plots the nodes in the list in the color specified as key - works for the colors known by nx.drawings
            (maybe same as for matplotlib?)
        """
        plt.figure()
        all_nodes = set(self.nodes)
        # use double loop structure to plot colored nodes after the blue ones - this way they show on top
        for color in coloring.keys():
            color_nodes = set(n for n in self.nodes if self.nodes[n]['index'] in coloring[color])
            all_nodes -= color_nodes
        nx.drawing.nx_pylab.draw_networkx(self, pos={n: self.nodes[n]['coordinate'][0:2] for n in self.nodes},
                                          nodelist=list(all_nodes))
        for color in coloring.keys():
            color_nodes = set(n for n in self.nodes if self.nodes[n]['index'] in coloring[color])
            nx.drawing.nx_pylab.draw_networkx(self, pos={n: self.nodes[n]['coordinate'][0:2] for n in self.nodes},
                                              nodelist=list(color_nodes), node_color=color)
        plt.show()

    def node_pos(self, *args, **kwargs):
        """
        Returns dict of node x-y positions for plots: {n:(x,y)}
        """
        return {n: (self.nodes[n]['coordinate'][0], self.nodes[n]['coordinate'][1]) for n in self.nodes}

class DnnData(object):
    def __init__(self, data_file, SE, n_train=0, n_val=0, d_dist=None, T_dist=None,
                 include_slack_input=False, include_slack_output=False):
        self.data_file = data_file
        self.start_ind_validation = n_train
        self.start_ind_test = n_train + n_val
        self.include_slack_input = include_slack_input
        self.include_slack_output = include_slack_output

        # Data shape specifications
        self.n_demands = len(SE.dem_order)
        self.n_supplies = len(SE.pp_order)
        n_actives = self.n_supplies + self.n_demands
        self.input_shape = [(None, self.n_demands), (None, self.n_supplies if include_slack_input else self.n_supplies-1),
                            (None, self.n_demands), (None, self.n_supplies)]
        self.output_shape = [(None, (SE.n_edges + SE.n_nodes)*2),
                             (None, n_actives if include_slack_output else n_actives - 1)]

        # store input-output samples as tensors
        self.train_inputs = None
        self.train_outputs = None
        self.validation_inputs = None
        self.validation_outputs = None
        self.test_inputs = None
        self.test_outputs = None

        # settings for data_generator:
        self.d_dist = d_dist
        self.T_dist = T_dist
        self.state_shape = 2*SE.n_nodes + 2*SE.n_edges

        # setup divider between supply and demand data

        self.dem_sign = SE.dem_sign
        self.temp_sign = SE.temp_sign
        self.dem_index_list = [ind for ind in SE.demands.keys() if self.dem_sign[SE.dem_ind[ind]] > 0]
        self.heat_index_list = [ind for ind in SE.demands.keys() if self.dem_sign[SE.dem_ind[ind]] < 0]
        self.heat_index_list.extend(SE.heatings.keys())

    def _load_data(self, n_samples, offset):
        [dq, T, state] = NN.import_training_data(n_samples, self.data_file, skip=offset, f_id0=0)
        # reorder data:
        d_vals = tf.gather(dq, np.where(self.dem_sign > 0)[0], axis=1)
        q_vals = tf.gather(dq, np.where(self.dem_sign < 0)[0], axis=1)
        T_d_vals = tf.gather(T, np.where(self.temp_sign > 0)[0], axis=1)
        T_q_vals = tf.gather(T, np.where(self.temp_sign < 0)[0], axis=1)

        # potentially remove slack input and output
        if not self.include_slack_input:
            if tf.shape(q_vals)[-1] == self.n_supplies:
                q_vals = q_vals[:, :-1]
        if not self.include_slack_output:
            if tf.shape(dq)[-1] == self.n_supplies+self.n_demands:
                dq = dq[:, :-1]

        # return input powers and feed in temperatures - true state and powers in original order
        return [d_vals, q_vals, T_d_vals, T_q_vals], [state, dq]

    def get_data(self, n_samples, section='train', adjust_validation_split=True):
        if n_samples == 0:
            warnings.warn('Requested 0 samples from data set. Returning None.')
            return [None], [None]

        if section == 'train':
            inputs = self.train_inputs
            outputs = self.train_outputs
            offset = 0
            if adjust_validation_split:
                self.start_ind_validation = max(self.start_ind_validation, n_samples)
        elif section == 'validation':
            inputs = self.validation_inputs
            outputs = self.validation_outputs
            offset = self.start_ind_validation
            self.start_ind_test = max(self.start_ind_test, offset + n_samples)
        elif section == 'test':
            inputs = self.test_inputs
            outputs = self.test_outputs
            offset = self.start_ind_test
        else:
            raise Exception(f'Unknown section specification {section}; supported sections are "train" or "validation"')

        n_available = 0 if inputs is None else inputs[0].shape()[0]
        if n_samples > n_available:
            try:
                new_inputs, new_outputs = self._load_data(n_samples-n_available, offset=offset)
            except NN.InsufficientDataError as e:
                raise NN.InsufficientDataError(task=f'load {section} data', skip=offset, **e.kwargs)
            if inputs is not None:
                for i, new in enumerate(new_inputs):
                    inputs[i] = tf.concat([inputs[i], new], axis=1)
            else:
                inputs = new_inputs
            if outputs is not None:
                for i, new in enumerate(new_outputs):
                    outputs[i] = tf.concat([outputs[i], new], axis=1)
            else:
                outputs = new_outputs

        return [inp[:n_samples, :] for inp in inputs], [out[:n_samples, :] for out in outputs]

    def get_data_generator(self, n_samples, batch_size):
        if self.d_dist is None or self.T_dist is None:
            raise Exception(f'pls specify "d_dist" and "T_dist" in order to use a data generator')
        return NN.InputGenerator(self.d_dist, self.T_dist, n_states=self.state_shape,
                                 n_samples_per_epoch=n_samples, batch_size=batch_size,
                                 include_slack_output=self.include_slack_output)

class UniformDistribution(tfd.Uniform):
    # def __init__(self, T_min, T_max, validate_args=False, allow_nan_stats=True, name='UniformDistribution'):
    #     super().__init__(low=T_min, high=T_max, validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name)
    #     self._parameters = dict(locals())    # store parameters for __repr__

    def __repr__(self):                    # overwrite __repr__ to include parameters
        return f"<state_estimation.se_lib.utility.UniformDistribution '{self.name}' " \
               f"batch_shape={self.batch_shape.as_list()} event_shape={self.event_shape.as_list()} " \
               f"low={self.low} high={self.high}>"

    # @classmethod
    # def _parameter_properties(cls, dtype, num_classes=None):
    #     # parameters have the same properties (e.g. shapes) as in non-truncated normal distributions
    #     return tfd.Uniform._parameter_properties(dtype, num_classes)

class ZeroTruncatedMultivariateNormal(tfd.MultivariateNormalTriL):
    '''
    adaptation of the MultivariateNormal distribution provided by tfd
    Cut off all values below zero; overwrite functions: sample, prob, log_prob accordingly
    '''
    def __init__(self, loc, scale_tril, validate_args=True, name='ZeroTruncatedMultivariateNormal'):
        parameters = dict(locals())
        self.signs = tf.math.sign(loc)
        self.sample_dim = loc.get_shape()[0]
        super().__init__(loc=tf.math.abs(loc), scale_tril=scale_tril, validate_args=validate_args, name=name)
        self._parameters = parameters

    def __repr__(self):
        return f"<state_estimation.se_lib.utility.ZeroTruncatedMultivariateNormal '{self.name}' " \
               f"batch_shape={self.batch_shape.as_list()} event_shape={self.event_shape.as_list()} " \
               f"d_type={self.dtype.__repr__()}>"

    def __str__(self):
        return f'state_estimation.se_lib.utility.ZeroTruncatedMultivariateNormal("{self.name}", ' \
               f'batch_shape={self.batch_shape.as_list()}, event_shape={self.event_shape.as_list()}, ' \
               f'd_type={self.dtype.__repr__()})'

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # parameters have the same properties (e.g. shapes) as in non-truncated normal distributions
        return tfd.MultivariateNormalTriL._parameter_properties(dtype, num_classes)

    def _mean(self):
        return self._parameters['loc']

    def sample(self, n_samples):
        # We override sample rather than _sample_n as advised in the tfp documentation.
        # Reasoning:
        #   the parent (tfd.MultivariateNormalTriL) overwrites _call_sample_n instead of the usual _sample_n
        #   therefore, self.sample() does is no longer redirected to _sample_n

        # initialise output tensor with all zeros
        init_samples = tf.zeros(shape=(n_samples, self.sample_dim), dtype=tf.float64)
        # check if any sample state is <0 (invalid sample) or =0 (not jet defined or removed due to production surplus)
        cond = lambda samples: tf.reduce_any(samples <= 0)

        # sample and fill valid rows into the output tensor
        def body(samples):
            new_samples = super(ZeroTruncatedMultivariateNormal, self).sample(n_samples)
            # replace samples that are not jet placed or which contain values below 0 with new samples
            #                  <overhead>  vvv actual cond. vvv    <    overhead to slice the right dimension    >
            samples = tf.where(tf.repeat(tf.reduce_all(samples > 0, axis=1, keepdims=True), self.sample_dim, axis=1),
                               samples, new_samples)

            # remove rows, which do not add up to values above zero (i.e. total demand is lower than production)
            samples = tf.where(tf.repeat(tf.reduce_sum(self.signs*samples, axis=1, keepdims=True) > 0,
                                         self.sample_dim, axis=1), samples, tf.zeros_like(samples))
            return [samples]

        [samples] = tf.while_loop(cond=cond, body=body, loop_vars=[init_samples])
        return samples * self.signs