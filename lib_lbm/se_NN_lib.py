import tensorflow as tf
import pandas as pd

import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from lib_lbm.utility.state_equations import StateEquations, calculate_physics_violation
from lib_lbm.classic_solver import SE_solver
import tensorflow_probability as tfp
tfd = tfp.distributions

#%% Data handling:

class InsufficientDataError(Exception):
    def __init__(self, message=None, task=None, n_avail=None, n_req=None, file_s=None, file_e=None, skip=None, **kwargs):
        self.kwargs = dict()
        if message is not None:
            self.kwargs['message'] = message
        self.task = task
        if task is not None:
            self.kwargs['task'] = task
        self.n_avail = n_avail
        if n_avail is not None:
            self.kwargs['n_avail'] = n_avail
        self.n_req = n_req
        if n_req is not None:
            self.kwargs['n_req'] = n_req
        self.file_s = file_s
        if file_s is not None:
            self.kwargs['file_s'] = file_s
        self.file_e = file_e
        if file_e is not None:
            self.kwargs['file_e'] = file_e
        self.skip = skip
        if skip is not None:
            self.kwargs['skip'] = skip

        task_snippet = f'for task: {self.task} ' if self.task is not None else ''
        file_snippet = f'in files {self.file_s} ... {self.file_e}' if self.file_s is not None and self.file_e is not None else ''
        request_snippet = f'Requested samples: {self.n_req}' if self.n_req is not None else ''
        avail_snippet = f'available: {self.n_avail} ' if self.n_avail is not None else ''
        skip_snippet = f' (skipped first {self.skip} samples)' if self.skip is not None else ''
        sep1 = ', ' if self.n_req is not None or self.n_avail is not None else ''
        sep2 = '; ' if self.n_req is not None and self.n_avail is not None else ''

        if message is None:
            message = f'Insufficient number of training samples ' \
                      f'{task_snippet}{file_snippet}{sep1}{request_snippet}{sep2}{avail_snippet}{skip_snippet}'
        super().__init__(message)

def import_training_data(n_samples, file_name, f_id0=1, skip=0, parse_weights=False, dtype=tf.float64):
    """
    n_samples: int
        number of samples to be loaded in
    file_name: string
        name of the files the data is read in from;
    f_id0: integer optional
        if the file is not found, add increasing integers at the end of the file name starting with f_id0
    skip: interger optional
        if not zero: skip the first n samples
    """
    def parse_line(line):
        line = line.translate({ord(']'): None})
        w, d, T, _, x = line.split('[')
        w = np.fromstring(w, dtype=np.float64, sep=' ') if not w=='' else 1
        d = np.fromstring(d, dtype=np.float64, sep=' ')
        T = np.fromstring(T, dtype=np.float64, sep=' ')
        x = np.fromstring(x, dtype=np.float64, sep=' ')
        return w, d, T, x

    def file_length(file_name):
        with open(file_name, 'r') as f:
            for i, _ in enumerate(f):
                pass
        return i+1

    def get_data_format(file_name):
        with open(file_name, 'r') as f:
            _, d, T, x = parse_line(f.readline())
            return d.shape, T.shape, x.shape
        
    def read_single_file_new(file_name, ws, ds, Ts, xs, start_index, end_index, skip):
        ind = start_index
        with open(file_name, 'r') as f:
            for i, l in enumerate(f):
                if i < skip:
                    continue
                else:
                    ws[ind], ds[ind,:], Ts[ind,:], xs[ind,:] = parse_line(l)
                    ind += 1
                    if ind == end_index:
                        break
        return ind

    try:
        data_format = get_data_format(file_name)
    except FileNotFoundError:
        if file_name.endswith('.csv'):
            file_name = file_name[:-4]
        if file_name.endswith('_'):
            file_name = file_name[:-1]
        data_format = get_data_format(f'{file_name}_{f_id0}.csv')
    ws = np.zeros((n_samples, 1))
    ds = np.zeros((n_samples, data_format[0][0]))
    Ts = np.zeros((n_samples, data_format[1][0]))
    xs = np.zeros((n_samples, data_format[2][0]))
    try:
        _ = read_single_file_new(file_name, ws, ds, Ts, xs, start_index=0, end_index=n_samples, skip=skip)
    except FileNotFoundError:
        f_id = f_id0
        n_read = 0
        start_index = 0
        while start_index != n_samples:
            f = f'{file_name}_{f_id}.csv'
            f_id += 1

            # skip files entirely if the number of lines is smaller than the skip value
            if skip > 0:
                f_len = file_length(f)
                if f_len <= skip:
                    # reduce skip by the number of skipped samples and continue with next file
                    skip -= f_len
                    continue

            # read single file returns last index, used as start_index in next iteration
            try:
                start_index = read_single_file_new(f, ws, ds, Ts, xs, start_index, n_samples, skip)
            except FileNotFoundError:
                raise InsufficientDataError(n_avail=start_index, n_req=n_samples,
                                            file_s=f'{file_name}_{f_id0}', file_e=f'{file_name}_{f_id-1}')
            skip = 0    # only skip lines in the first file read

    if parse_weights:
        return [tf.constant(val, dtype=dtype) for val in [ws, ds, Ts, xs]]
    else:
        return [tf.constant(val, dtype=dtype) for val in [ds, Ts, xs]]


class InputGenerator(keras.utils.Sequence):
    def __init__(self, q_distribution, T_distribution, n_states, n_samples_per_epoch, batch_size=32,
                 include_slack_output=False):
        self.q_distribution = q_distribution
        self.T_distribution = T_distribution
        self.batch_size = batch_size
        dem_sign = tf.sign(q_distribution.mean())
        temp_sign = np.pad(dem_sign, (0, 1), constant_values=-1)
        self.dem_ind = tf.squeeze(tf.where(dem_sign > 0))
        self.heat_ind = tf.squeeze(tf.where(dem_sign < 0))
        self.heat_ind_T = tf.squeeze(tf.where(temp_sign < 0))
        self.output = lambda n_samples: tf.ones((n_samples, n_states), dtype=q_distribution.mean().dtype)
        self.n_samples = n_samples_per_epoch
        self.include_slack_output = include_slack_output

    def on_epoch_end(self):
        # do nothing at the end of each epoch (no point to shuffle here)
        pass

    def __len__(self):
        """ Denotes the number of batches per epoch """
        return int(np.floor(self.n_samples / self.batch_size))

    @tf.function
    def __data_generation(self, n_samples):
        q_vals = self.q_distribution.sample(n_samples)

        q_vals_dem = tf.gather(q_vals, self.dem_ind, axis=1)
        q_vals_heat = tf.gather(q_vals, self.heat_ind, axis=1)
        if self.include_slack_output:
            q_vals = tf.pad(q_vals, [[0, 0], [0, 1]], mode='constant', constant_values=-1.)
        T_vals = self.T_distribution.sample(n_samples)
        T_vals_dem = tf.gather(T_vals, self.dem_ind, axis=1)
        T_vals_heat = tf.gather(T_vals, self.heat_ind_T, axis=1)

        # reshape to (n_samples, -1) for consistently with the output shape between single indices and multiple indices
        return tuple(tf.reshape(output, (n_samples, -1)) for output in (q_vals, q_vals_dem, q_vals_heat, T_vals_dem, T_vals_heat))

    def generate_data(self, n_samples):
        q_vals, q_vals_dem, q_vals_heat, T_vals_dem, T_vals_heat = self.__data_generation(n_samples)
        input = (q_vals_dem, q_vals_heat, T_vals_dem, T_vals_heat)
        output = (self.output(n_samples), q_vals)
        return input, output

    def __getitem__(self, _):
        """ Generate one batch of data """
        return self.generate_data(self.batch_size)

    def __call__(self):
        # standard code for calling data generator
        for i in range(self.__len__()):
            yield self.__getitem__(i)

            if i == self.__len__() - 1:
                self.on_epoch_end()

#%% Custom Layers:
class MyScalingLayer(keras.layers.Layer):
    def __init__(self, offset=None, scaling=None, mapping_matrix=None, mapping_matrix_ind=None,
                 mapping_matrix_dense_shape=None, name='MyScalingLayer'):
        '''
        output = offset + mapping_matrix @ scaling * inputs

        '''
        super().__init__(name=name)
        if offset is not None:
            self.offset = tf.Variable(offset, trainable=False)
        if scaling is not None:
            self.scaling = tf.Variable(scaling, trainable=False)
        if mapping_matrix is not None:
            self.mapping_matrix = mapping_matrix
        if mapping_matrix_ind is not None:
            self.mapping_matrix = tf.sparse.SparseTensor(indices=mapping_matrix_ind,
                                                         values=tf.ones(tf.shape(mapping_matrix_ind)[0],
                                                                        dtype=tf.float64),
                                                         dense_shape=mapping_matrix_dense_shape)
            self.offset = tf.Variable(tf.ones((mapping_matrix_dense_shape[0], 1), dtype=tf.float64), trainable=False)
            self.scaling = tf.Variable(tf.ones(tf.shape(mapping_matrix_ind)[0], dtype=tf.float64), trainable=False)

    def call(self, inputs, *args, **keyargs):
        return tf.transpose(tf.math.add(self.offset,
                                        tf.sparse.sparse_dense_matmul(self.mapping_matrix.with_values(self.scaling),
                                                                      inputs, adjoint_b=True)))

    def get_config(self):
        return {
            'name': self.name,
            'mapping_matrix_ind': tf.cast(self.mapping_matrix.indices, tf.int64).numpy(),
            'mapping_matrix_dense_shape': tf.cast(self.mapping_matrix.shape, tf.int64).numpy()
            }


class custom_add(keras.layers.Layer):
    # for some reason keras.Layers.add does not have a get_config with the tensorflow version used in this project
    def __init__(self, name='add_layer'):
        super().__init__(name=name)

    def call(self, inputs, *args, **kwargs):
        return tf.reduce_sum(inputs, axis=0)

    def get_config(self):
        return {
            'name': self.name
            }


class get_demands(keras.layers.Layer):
    """ calculates the demand based on a state vector """
    def __init__(self, SE=None, add_slack_output=False, Ts=None, Tr=None, mf=None,  name='demands'):
        super().__init__(name=name)
        if SE is not None:
            n_dem = len(SE.demands.keys()) +1 if add_slack_output else len(SE.demands.keys())
            self.Ts_pos = np.zeros(n_dem, dtype=int)
            self.Tr_pos = np.zeros(n_dem, dtype=int)
            self.mf_pos = np.zeros(n_dem, dtype=int)
            for dem in SE.demands.keys():
                ind = SE.dem_ind[dem]
                dem_pos = SE.find_edge[dem]
                self.Ts_pos[ind] = SE.find_node[SE.edges[dem_pos]['from']]
                self.Tr_pos[ind] = SE.find_node[SE.edges[dem_pos]['to']]
                self.mf_pos[ind] = SE.n_nodes + dem_pos
            if add_slack_output:
                ind = -1
                heat_pos = SE.find_edge[list(SE.heatings.keys())[0]]
                self.Ts_pos[ind] = SE.find_node[SE.edges[heat_pos]['from']]
                self.Tr_pos[ind] = SE.find_node[SE.edges[heat_pos]['to']]
                self.mf_pos[ind] = SE.n_nodes + heat_pos
        else:
            self.Ts_pos = Ts
            self.Tr_pos = Tr
            self.mf_pos = mf
        self.add_slack_output = add_slack_output
        self.cp = tf.constant(4.180, name='cp_water_in_kJ/kgK', dtype=tf.float64)

    @tf.function()
    def call(self, inputs, *args, **kwargs):
        Ts = tf.gather(inputs, self.Ts_pos, axis=-1)
        Tr = tf.gather(inputs, self.Tr_pos, axis=-1)
        mf = tf.gather(inputs, self.mf_pos, axis=-1)
        Q_heat = mf * self.cp * (Ts - Tr)
        return Q_heat

    def get_config(self):
        return {
            'name': self.name,
            'Ts': self.Ts_pos,
            'Tr': self.Tr_pos,
            'mf': self.mf_pos,
            'add_slack_output': self.add_slack_output
        }

def build_DNN(SE, f_index, c_index,
              q_f, q_c, T_f, T_c, states=None, normalise_output_on_data=True, d_prior_dist=None, T_prior_dist=None,
              layer_spec=[10, 10], add_demands_output=False, add_slack_output=False,
              cycles=None):
    """
        returns a DNN expecting four inputs: d, q, T_d, T_q;
            and up to two outputs: states, heat powers (last one only if add_demands_output=True)

        function inputs:
            SE: state estimation object
            f_index: list of str, indices of fixed power values in q_f, i.e. heat demands
            c_index: list of str, indices of controllable power values in q_c, i.e. heat supplies
            d, q, T_d, T_q, states: training data used for normalisation and shape interpolation
            normalise_output_on_data: bool, if True, the output is normalised based on the training data
            else: normalised based on the first order Taylor expansion of the state mapping
            d_prior_dist: tfp.distributions object, prior distribution for the demand
            T_prior_dist: tfp.distributions object, prior distribution for the temperatures
            layer_spec: list of int, specifying the number of nodes in each hidden layer
            add_demands_output: bool, if True, the DNN returns the heat demands as well
            add_slack_output: bool, if True, the DNN returns the slack heat demand as well
            cycles: list specifying the cycles in the heat grid
    """

    n_nodes = SE.n_nodes
    n_edges = SE.n_edges
    n_states = 2 * n_nodes + 2 * n_edges  # number of states in the grid
    n_real_outputs = tf.shape(SE.mask_matrix_full)[1]  # number of states that are not a priori fixed
    n_actives = SE.n_active_edges  # all active edges -> consumer or producer
    n_q_f = tf.shape(q_f).numpy()[-1]  # number of power values fixed power values
    n_q_c = tf.shape(q_c).numpy()[-1]  # number of power values controllable power values
    n_T_f = tf.shape(T_f).numpy()[-1]  # number of temperatures for fixed power values
    n_T_c = tf.shape(T_c).numpy()[-1]  # number of temperatures for controllable power values

    # scaling layers in the DNN: scale inputs to [0, 1], outputs from N(0,1) to real size (including zero-mappings)
    def generate_01_layer(data, name='', set_floor_zero=False):
        # returns a layer scaling the input data to [0; 1]
        if not set_floor_zero:
            min_val = tf.reduce_min(tf.math.abs(data), axis=0)
        else:
            min_val = tf.zeros_like(data[0, :])
        max_val = tf.reduce_max(tf.math.abs(data), axis=0) * tf.math.sign(data[0, :])
        slope = tf.math.divide_no_nan(tf.ones_like(max_val), (max_val - min_val))
        slope = tf.where(slope == 0, 1, slope)
        offset = tf.expand_dims(- min_val * slope, axis=-1)
        mapping = tf.sparse.eye(tf.shape(data)[-1])
        # output = slope * input + offset
        return MyScalingLayer(offset=offset, scaling=slope, mapping_matrix=mapping, name=name)

    # input scaling layers
    qc_scaling = generate_01_layer(q_c, name='q_scaling', set_floor_zero=True)
    qf_scaling = generate_01_layer(q_f, name='d_scaling', set_floor_zero=True)
    Tc_scaling = generate_01_layer(T_c, name='T_q_scaling', set_floor_zero=False)
    Tf_scaling = generate_01_layer(T_f, name='T_d_scaling', set_floor_zero=False)

    # prepare output scaling layers:
    if states is not None and normalise_output_on_data:
        state_mean = tf.transpose(tf.reduce_mean(states, axis=0, keepdims=True))
        state_std = tf.math.reduce_std(states, axis=0)
        # scaling_std: scale parameter used for result upscaling; remove vector to "real output values" determined by
        # the mask matrix of the SE object
        scaling_std = state_std[tf.reduce_any(tf.sparse.to_dense(SE.mask_matrix_full) == 1, axis=1)]
    else:
        # solve state equations for mean conditions:
        solve_SE = SE_solver()
        dem = d_prior_dist.mean()
        Temp = T_prior_dist.mean()
        SE.load_save_state()
        for dem in SE.demands.keys():
            ind = SE.dem_ind[dem]
            SE.set_active_edge_temperature(dem, Temp[ind])
        # set temperature for heating - last entry in T_vector
        e_ind = [k for k in SE.heatings.keys()][0]
        SE.set_active_edge_temperature(e_ind, Temp[-1])
        solve_SE(SE, cycles=cycles, verbose=True)
        state_mean = tf.concat([SE.T, SE.mf, SE.p, SE.T_end], axis=0)
        J = SE.evaluate_state_equations('demand jacobian')
        prior_state_cov = J @ d_prior_dist.covariance() @ tf.transpose(J)
        state_std = tf.Variable(tf.math.sqrt(tf.linalg.diag_part(prior_state_cov)))
        # scaling_std: scale parameter used for result upscaling; No values have to be removed, as the jacobian
        # already only considers "real outputs"
        scaling_std = state_std

    # some temperatures solely depend on the DNN inputs (T_end actives, T_node at output actives);
    # set scaling mean to zero and add entry to T-mapping matrix to assign the corresponding temperature input
    T_f_scaling_matrix = np.zeros([n_states, n_T_f])
    T_c_scaling_matrix = np.zeros([n_states, n_T_c])
    zero_indices = []

    for i, ind in enumerate(f_index):
        e_ind = SE.find_edge[ind]
        n_ind = SE.find_node[SE.edges[e_ind]['to']]
        T_f_scaling_matrix[n_ind, i] = 1
        T_f_scaling_matrix[2 * n_nodes + n_edges + e_ind, i] = 1
        zero_indices.append(n_ind)
        zero_indices.append(2 * n_nodes + n_edges + e_ind)

    for i, ind in enumerate(c_index):
        e_ind = SE.find_edge[ind]
        n_ind = SE.find_node[SE.edges[e_ind]['to']]
        T_c_scaling_matrix[n_ind, i] = 1
        T_c_scaling_matrix[2 * n_nodes + n_edges + e_ind, i] = 1
        zero_indices.append(n_ind)
        zero_indices.append(2 * n_nodes + n_edges + e_ind)

    state_mean = tf.tensor_scatter_nd_update(state_mean, [[i] for i in zero_indices],
                                             [[0] for _ in zero_indices])

    # output scaling layers & temperature mappings:
    state_scale = MyScalingLayer(offset=tf.constant(state_mean), scaling=scaling_std,
                                 mapping_matrix=SE.mask_matrix_full, name='rescale_outputs')
    temp_f_assign = MyScalingLayer(offset=tf.zeros_like(state_mean),
                                   scaling=tf.ones(shape=(2 * n_T_f,), dtype=tf.float64),
                                   mapping_matrix=tf.sparse.from_dense(T_f_scaling_matrix), name='temp_assignment_f')
    temp_c_assign = MyScalingLayer(offset=tf.zeros_like(state_mean),
                                   scaling=tf.ones(shape=(2 * n_T_c,), dtype=tf.float64),
                                   mapping_matrix=tf.sparse.from_dense(T_c_scaling_matrix), name='temp_assignment_c')

    """-------------------- Build DNN: ---------------------"""
    # inputs:
    inputs_qf = keras.Input(shape=(n_q_f,), name='d')
    inputs_qc = keras.Input(shape=(n_q_c,), name='q')
    inputs_Tf = keras.Input(shape=(n_T_f,), name='t_d')
    inputs_Tc = keras.Input(shape=(n_T_c,), name='t_q')
    # scaling:
    q_f_scaled = qf_scaling(inputs_qf)
    q_c_scaled = qc_scaling(inputs_qc)
    T_f_scaled = Tf_scaling(inputs_Tf)
    T_c_scaled = Tc_scaling(inputs_Tc)
    # concatenate and ReLus:
    x = keras.layers.concatenate([q_f_scaled, q_c_scaled, T_f_scaled, T_c_scaled], name='concat_input')
    i = 0
    for i, n in enumerate(layer_spec):
        x = keras.layers.Dense(n, activation='relu', name=f'ReLU_{i}')(x)
    x = keras.layers.Dense(n_real_outputs, activation='linear', name=f'Linear_{i + 1}')(x)
    # rescale DNN predictions and insert temperatures
    state_outputs = state_scale(x)
    temp_f_outputs = temp_f_assign(inputs_Tf)
    temp_c_outputs = temp_c_assign(inputs_Tc)
    pred_states = custom_add(name='Grid_state')([state_outputs, temp_f_outputs, temp_c_outputs])

    if add_demands_output:
        demands = get_demands(SE, add_slack_output)(pred_states)
        return keras.Model(inputs=[inputs_qf, inputs_qc, inputs_Tf, inputs_Tc], outputs=[pred_states, demands],
                           name='DtoS_model')
    else:
        return keras.Model(inputs=[inputs_qf, inputs_qc, inputs_Tf, inputs_Tc], outputs=[pred_states],
                           name='DtoS_model')

#%% loss functions:
@tf.function
def loss_SE_new(SE, y_true, y_pred):
    # calculations are way easier to get consistent if the batch dimension is last:
    loss = SE.calculate_physics_violation(y_pred)
    return tf.reduce_sum(loss ** 2, axis=0)


@tf.function
def loss_SE(SE, T, mf, p, T_end):
    # calculations are way easier to get consistent if the batch dimension is last:
    loss = SE.evaluate_state_equations('forwardpass',
                                       T=tf.transpose(T), mf=tf.transpose(mf), p=tf.transpose(p),
                                       T_end=tf.transpose(T_end))
    return tf.reduce_sum(loss ** 2, axis=0)


@tf.function
def loss_measurement(measurement, prediction, measurement_noise):
    rv_y = tfd.MultivariateNormalDiag(loc=measurement, scale_diag=measurement_noise)
    return -rv_y.log_prob(prediction)


@tf.function
def loss_prior_demand(SE, T, mf, T_end, prior, dem_edg_ind, dem_f_node_ind):
    # calculate the  predicted demands from the states and compares with the prior assumption
    def predict_demand(i):
        return SE.eq_demand_Q(mf[..., dem_edg_ind[i]], 0, T[..., dem_f_node_ind[i]], T_end[..., dem_edg_ind[i]])
    pred_demands = tf.transpose(
        tf.map_fn(predict_demand, tf.range(tf.shape(dem_edg_ind)[0]), fn_output_signature=tf.float64))

    return -prior.log_prob(pred_demands)


@tf.function
def loss_mape(y_true, y_pred):
    return 100. * tf.reduce_mean(tf.math.abs((y_true - y_pred) / y_true))


@tf.function
def loss_mae(y_true, y_pred):
    return tf.reduce_mean(tf.math.abs(y_true - y_pred))

#%% Loss Classes based on the loss functions

class LossSE(keras.losses.Loss):
    def __init__(self, SE=None, config=None, name='SE_loss', run_eager=False):
        super().__init__(name=name)
        # duplicate SE-object; this allows to retrace internal functions with different parameters, i.e. without demands
        if SE is not None:
            self.config = SE.get_phyical_loss_config()
        else:
            config['A_red_dense'] = tf.constant(config['A_red_dense'])
            config['A_dense'] = tf.constant(config['A_dense'])
            config['B_mask_matrix_dense'] = tf.constant(config['B_mask_matrix_dense'])
            config['Ta'] = tf.constant(config['Ta'])
            config['cp'] = tf.constant(config['cp'])
            config['rho'] = tf.constant(config['rho'])
            self.config = config
        self.n_nodes = self.config['n_nodes']
        self.n_edges = self.config['n_edges']
        self.class_name = 'LossSE'
        self.run_eager = run_eager
        
        @tf.function
        def loss_function(y_pred):
            loss = calculate_physics_violation(y_pred=tf.expand_dims(y_pred, axis=1), **self.config)
            return loss

        self.concrete_loss_function = loss_function.get_concrete_function(tf.ones(shape=(2*self.n_nodes+2*self.n_edges,), dtype=tf.float64))
        # self.concrete_loss_function = loss_function

    @tf.function
    def call(self, y_true, y_pred):
        """ y_true -> void, y_pred: predicted state, returns left hand side of  state equations """
        lse = tf.vectorized_map(self.concrete_loss_function, elems=y_pred)
        # lse = [self.concrete_loss_function(y) for y in y_pred]
        return tf.reduce_mean(lse)

    def get_config(self):
        config = self.config.copy()
        config['A_red_dense'] = config['A_red_dense'].numpy()
        config['A_dense'] = config['A_dense'].numpy()
        config['B_mask_matrix_dense'] = config['B_mask_matrix_dense'].numpy()
        config['Ta'] = config['Ta'].numpy()
        config['cp'] = config['cp'].numpy()
        config['rho'] = config['rho'].numpy()

        return {'SE_config': config,
                'name': self.name}


class LossPower(keras.losses.Loss):
    def __init__(self, y_shape, mask_slack, name='Power_loss'):
        self.y_shape = y_shape
        self.mask_slack = mask_slack

        self.loss_mask = tf.ones(shape=(1, y_shape[1]), dtype=tf.float64)
        if mask_slack:
            self.loss_mask = tf.tensor_scatter_nd_update(self.loss_mask, [[0, tf.shape(self.loss_mask)[1]-1]], [0.])
        # self.loss_mask = tf.ones(shape=(1, 7))
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        return tf.reduce_mean(self.loss_mask * tf.math.square(y_true - y_pred), axis=-1)

    def get_config(self):
        return {'y_shape': self.y_shape,
                'mask_slack': self.mask_slack,
                'name': self.name}


class LossSE_eager(keras.losses.Loss):
    def __init__(self, SE=None, SE_config=None, name='SE_loss'):
        super().__init__(name=name)
        # duplicate SE-object; this allows to retrace internal functions with different parameters, i.e. without demands
        if SE is not None:
            config = SE.get_config()
        else:
            config = SE_config
        config['use_demands'] = False
        self.SE_config = config
        self.SE = StateEquations(**config)
        self.n_nodes = self.SE.n_nodes
        self.n_edges = self.SE.n_edges
        self.class_name = 'LossSE'

    def call(self, y_true, y_pred):
        """ y_true -> void, y_pred: predicted state, returns left hand side of  state equations """
        [T, mf, p, T_end] = tf.split(y_pred, num_or_size_splits=[self.n_nodes, self.n_edges,
                                                                 self.n_nodes, self.n_edges], axis=-1)
        # lse = [loss_SE(self.SE, T[i:i+1,:], mf[i:i+1,:], p[i:i+1,:], T_end[i:i+1,:]) for i in range(T.get_shape()[0])]
        for y_pred_i in y_pred:
            lse = loss_SE(self.SE, y_true, y_pred_i)
            lse = loss_SE(self.SE, y_true, y_pred_i)
            lse = loss_SE(self.SE, y_true, y_pred_i)
            lse = loss_SE(self.SE, y_true, y_pred_i)
        return tf.reduce_mean(lse)

    def get_config(self):
        return {'SE_config': self.SE_config,
                'name': self.name}


class LossMeasurement(keras.losses.Loss):
    def __init__(self, measurement_index, noise_scale, name='measurement_loss'):
        super().__init__(name=name)
        self.measurement_index = measurement_index
        self.noise_scale = noise_scale

    def call(self, y_true, y_pred):
        """ y_true -> actual measurement values, y_pred: predicted state; returns -log_prob (y_pred|y_true) """
        pred_measurements = tf.gather(y_pred, self.measurement_index, axis=-1)
        return loss_measurement(y_true, pred_measurements, self.noise_scale)


class LossPriorDemands(keras.losses.Loss):
    def __init__(self, SE, prior, n_nodes, n_edges, name='prior_demand_loss'):
        super().__init__(name=name)
        self.SE = SE
        self.prior = prior
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        dem_edg_ind = []
        dem_f_node_ind = []
        for d in SE.prior_demands.keys():
            ind = (SE.find_edge[d])
            dem_edg_ind.append(ind)
            dem_f_node_ind.append(SE.find_node[SE.edges[ind]['from']])
        self.dem_edg_ind = tf.constant(dem_edg_ind)
        self.dem_f_node_ind = tf.constant(dem_f_node_ind)
        self.class_name = 'LossPriorDemand'

    def call(self, y_true, y_pred):
        """ y_true -> void, y_pred: predicted state; returns -log_prob (y_pred|prior_demand) """
        [T, mf, p, T_end] = tf.split(y_pred, num_or_size_splits=[self.n_nodes, self.n_edges,
                                                                 self.n_nodes, self.n_edges], axis=-1)
        return loss_prior_demand(self.SE, T, mf, T_end, self.prior, self.dem_edg_ind, self.dem_f_node_ind)


class LossCombined(keras.losses.Loss):
    def __init__(self, SE, measurement_index, noise_scale, prior, n_nodes, n_edges, lambdaSE=1, lambdaM=1, lambdaP=1,
                 name='combined_loss'):
        super().__init__(name=name)
        self.l_SE = LossSE(SE, n_nodes, n_edges)
        self.l_M = LossMeasurement(measurement_index, noise_scale)
        self.l_P = LossPriorDemands(SE, prior)
        self.lambdaSE = lambdaSE
        self.lambdaM = lambdaM
        self.lambdaP = lambdaP
        self.class_name = 'LossCombined'

    def call(self, y_true, y_pred):
        """ y_true -> actual measurement values; y_pred: predicted state; returns weighted sum of losses"""
        return self.lambdaSE * self.l_SE.call(None, y_pred) + \
               self.lambdaM * self.l_M.call(y_true, y_pred) + \
               self.lambdaP * self.l_P.call(None, y_pred)


class LossPenalty(keras.losses.Loss):
    def __init__(self, pen_std, n_dims, name='penalty_for_unscaled_outputs'):
        super().__init__(name=name)
        self.dist = tfd.MultivariateNormalDiag(loc=tf.zeros(n_dims, tf.float64),
                                               scale_diag=(pen_std * tf.ones(n_dims, dtype=tf.float64)))

    def call(self, y_true, y_pred):
        """ y_true -> void, y_pred: value of dim. n_dims, scaled to zero mean and unit variance of
            returns -log_prob (y_prd|N(0, pen_std^2)
            use as a penalty for to large deviations
        """
        return -self.dist.log_prob(y_pred)


class LossWeightedMSE(keras.losses.Loss):
    def __init__(self, n_nodes, n_edges, lambda_T=1, lambda_mf=1, lambda_p=1, lambda_Tend=1, name='weighted_MSE'):
        super().__init__(name=name)
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.lambda_T = lambda_T
        self.lambda_mf = lambda_mf
        self.lambda_p = lambda_p
        self.lambda_Tend = lambda_Tend
        self.class_name = 'LossWeightedMSE'
        weighting_vector = np.zeros((2*n_edges+2*n_nodes))
        weighting_vector[0: n_nodes] = lambda_T
        weighting_vector[n_nodes: n_nodes+n_edges] = lambda_mf
        weighting_vector[n_nodes+n_edges: 2*n_nodes+n_edges] = lambda_p
        weighting_vector[2*n_nodes+n_edges: 2*n_nodes+2*n_edges] = lambda_Tend
        self.weighting_vector = tf.constant(weighting_vector)

    def call(self, y_true, y_pred):
        """ calculates the mean squared distance between y_true and y_pred, weighting T, mf, p, Tend with cor. lambda"""
        return tf.reduce_mean(self.weighting_vector*(y_true - y_pred)**2, axis=1)

    def get_config(self):
        return{'n_nodes': self.n_nodes,
               'n_edges': self.n_edges,
               'lambda_T': self.lambda_T,
               'lambda_mf': self.lambda_mf,
               'lambda_p': self.lambda_p,
               'lambda_Tend': self.lambda_Tend,
               'name': self.name}


class LossWeightedCombinedLoss(keras.losses.Loss):
    def __init__(self, losses=None, weights=None, normalise_weights=True, name='combined_loss',
                 loss_names=None, losses_configs=None):
        super().__init__(name=name)

        loss_mapping = {
            'LossSE': LossSE,
            'LossWeightedMSE': LossWeightedMSE
        }
        if losses is not None:
            self.losses = losses
        else:
            losses = [loss_mapping[l_str](**l_config) for l_str, l_config in zip(loss_names, losses_configs)]
        self.normalise_weights = normalise_weights
        if normalise_weights:
            self.weights = tf.constant(weights, dtype=tf.float64)/sum(weights)
        else:
            self.weights = tf.constant(weights, dtype=tf.float64)

        @tf.function
        def loss_function(y_true, y_pred, losses, weights):
            l = 0
            for i in range(len(losses)):
                l += weights[i] * losses[i](y_true, y_pred)
            return l
        self.loss_function_init = lambda y_true, y_pred: loss_function(y_true, y_pred, losses, weights)

    def call(self, y_true, y_pred):
        return self.loss_function_init(y_true, y_pred)

    def get_config(self):
        return {
            'loss_names': [L.class_name for L in self.losses],
            'losses_configs': [L.get_config() for L in self.losses],
            'weights': self.weights.numpy(),
            'normalise_weights': self.normalise_weights,
            'name': self.name
        }

#%% Metrics - Similar Objects to Loss-classes but slightly other structure
class CustomMetric(keras.metrics.Metric):
    """
    parent class to reduce overhead of custom metrics
    sub-classes only define the __init__ and the update_state functions
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = tf.Variable(0.0, dtype=tf.float64, name='current_metric_value')

    def result(self):
        return self.value

    def reset_state(self):
        self.value.assign(0.0)


class MetricSE(CustomMetric):
    def __init__(self, SE, name='SE_Metric'):
        super().__init__(name=name)
        config = SE.get_config()
        config['use_demands'] = False
        self.SE = StateEquations(**config)
        self.value = tf.Variable(0.0, dtype=tf.float64, name='SE_value')
        self.n_nodes = SE.n_nodes
        self.n_edges = SE.n_edges

    def update_state(self, y_true, y_pred, sample_weight=None):

        lse = tf.map_fn(lambda y_pred: loss_SE(self.SE, y_true, y_pred),
                        elems=y_pred, fn_output_signature=tf.float64)
        self.value.assign(tf.reduce_mean(lse))


class MetricMeasurement(CustomMetric):
    def __init__(self, measurement_index, noise_scale, name='Measurement_Metric'):
        super().__init__(name=name)
        self.measurement_index = measurement_index
        self.noise_scale = noise_scale
        self.value = tf.Variable(0.0, dtype=tf.float64, name='Measurement_loss_value')

    def update_state(self, y_true, y_pred, sample_weight=None):
        pred_measurements = tf.gather(y_pred, self.measurement_index, axis=-1)
        l = loss_measurement(y_true, pred_measurements, self.noise_scale)
        value = tf.cast(l, tf.float64)
        self.value.assign(tf.reduce_mean(value))


class MetricPriorDemands(CustomMetric):
    def __init__(self, SE, prior, n_nodes, n_edges, name='Prior_Demand_Metric'):
        super().__init__(name=name)
        self.SE = SE
        self.prior = prior
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        dem_edg_ind = []
        dem_f_node_ind = []
        for d in SE.prior_demands.keys():
            ind = (SE.find_edge[d])
            dem_edg_ind.append(ind)
            dem_f_node_ind.append(SE.find_node[SE.edges[ind]['from']])
        self.dem_edg_ind = tf.constant(dem_edg_ind)
        self.dem_f_node_ind = tf.constant(dem_f_node_ind)
        self.value = tf.Variable(0.0, dtype=tf.float64, name='SE_value')

    def update_state(self, y_true, y_pred, sample_weight=None):
        [T, mf, p, T_end] = tf.split(y_pred, num_or_size_splits=[self.n_nodes, self.n_edges,
                                                                 self.n_nodes, self.n_edges], axis=-1)
        l = loss_prior_demand(self.SE, T, mf, T_end, self.prior, self.dem_edg_ind, self.dem_f_node_ind)
        self.value.assign(tf.reduce_sum(tf.math.abs(l)))



class MetricMAPE_T(CustomMetric):
    def __init__(self, n_nodes, n_edges, name='MAPE_T', **kwargs):
        super().__init__(name=name)
        self.lb = 0
        self.ub = n_nodes
        self.value = tf.Variable(0.0, dtype=tf.float64, name='SE_value')
        self.n_nodes = n_nodes
        self.n_edges = n_edges

    def update_state(self, y_true, y_pred, sample_weight=None):
        l = loss_mape(y_pred=y_pred[:, self.lb:self.ub], y_true=y_true[:, self.lb:self.ub])
        self.value.assign(l)

    def get_config(self):
        return {
            'n_nodes': self.n_nodes,
            'n_edges': self.n_edges,
            'name': self.name
        }


class MetricMAPE_mf(CustomMetric):
    def __init__(self, n_nodes, n_edges, name='MAPE_mf', **kwargs):
        super().__init__(name=name)
        self.lb = n_nodes
        self.ub = n_nodes + n_edges
        self.value = tf.Variable(0.0, dtype=tf.float64, name='SE_value')
        self.n_nodes = n_nodes
        self.n_edges = n_edges

    def update_state(self, y_true, y_pred, sample_weight=None):
        l = loss_mape(y_pred=y_pred[:, self.lb:self.ub], y_true=y_true[:, self.lb:self.ub])
        self.value.assign(l)

    def get_config(self):
        return {
            'n_nodes': self.n_nodes,
            'n_edges': self.n_edges,
            'name': self.name
        }


class MetricMAPE_p(CustomMetric):
    def __init__(self, n_nodes, n_edges, name='MAPE_p', **kwargs):
        super().__init__(name=name)
        self.lb = n_nodes + n_edges
        self.ub = 2 * n_nodes + n_edges
        self.value = tf.Variable(0.0, dtype=tf.float64, name='SE_value')
        self.n_nodes = n_nodes
        self.n_edges = n_edges

    def update_state(self, y_true, y_pred, sample_weight=None):
        l = loss_mape(y_pred=y_pred[:, self.lb:self.ub], y_true=y_true[:, self.lb:self.ub])
        self.value.assign(l)

    def get_config(self):
        return {
            'n_nodes': self.n_nodes,
            'n_edges': self.n_edges,
            'name': self.name
        }


class MetricMAPE_Tend(CustomMetric):
    def __init__(self, n_nodes, n_edges, name='MAPE_Tend', **kwargs):
        super().__init__(name=name)
        self.lb = 2 * n_nodes + n_edges
        self.ub = 2 * n_nodes + 2 * n_edges
        self.value = tf.Variable(0.0, dtype=tf.float64, name='SE_value')
        self.n_nodes = n_nodes
        self.n_edges = n_edges

    def update_state(self, y_true, y_pred, sample_weight=None):
        l = loss_mape(y_pred=y_pred[:, self.lb:self.ub], y_true=y_true[:, self.lb:self.ub])
        self.value.assign(l)

    def get_config(self):
        return {
            'n_nodes': self.n_nodes,
            'n_edges': self.n_edges,
            'name': self.name
        }


class MetricMAE_T(CustomMetric):
    def __init__(self, n_nodes, n_edges, name='MAPE_T', **kwargs):
        super().__init__(name=name)
        self.lb = 0
        self.ub = n_nodes
        self.value = tf.Variable(0.0, dtype=tf.float64, name='SE_value')
        self.n_nodes = n_nodes
        self.n_edges = n_edges

    def update_state(self, y_true, y_pred, sample_weight=None):
        l = loss_mae(y_pred=y_pred[:, self.lb:self.ub], y_true=y_true[:, self.lb:self.ub])
        self.value.assign(l)

    def get_config(self):
        return {
            'n_nodes': self.n_nodes,
            'n_edges': self.n_edges,
            'name': self.name
        }


class MetricMAE_mf(CustomMetric):
    def __init__(self, n_nodes, n_edges, name='MAPE_mf', **kwargs):
        super().__init__(name=name)
        self.lb = n_nodes
        self.ub = n_nodes + n_edges
        self.value = tf.Variable(0.0, dtype=tf.float64, name='SE_value')
        self.n_nodes = n_nodes
        self.n_edges = n_edges

    def update_state(self, y_true, y_pred, sample_weight=None):
        l = loss_mae(y_pred=y_pred[:, self.lb:self.ub], y_true=y_true[:, self.lb:self.ub])
        self.value.assign(l)

    def get_config(self):
        return {
            'n_nodes': self.n_nodes,
            'n_edges': self.n_edges,
            'name': self.name
        }


class MetricMAE_p(CustomMetric):
    def __init__(self, n_nodes, n_edges, name='MAPE_p', **kwargs):
        super().__init__(name=name)
        self.lb = n_nodes + n_edges
        self.ub = 2 * n_nodes + n_edges
        self.value = tf.Variable(0.0, dtype=tf.float64, name='SE_value')
        self.n_nodes = n_nodes
        self.n_edges = n_edges

    def update_state(self, y_true, y_pred, sample_weight=None):
        l = loss_mae(y_pred=y_pred[:, self.lb:self.ub], y_true=y_true[:, self.lb:self.ub])
        self.value.assign(l)

    def get_config(self):
        return {
            'n_nodes': self.n_nodes,
            'n_edges': self.n_edges,
            'name': self.name
        }


class MetricMAE_Tend(CustomMetric):
    def __init__(self, n_nodes, n_edges, name='MAPE_Tend', **kwargs):
        super().__init__(name=name)
        self.lb = 2 * n_nodes + n_edges
        self.ub = 2 * n_nodes + 2 * n_edges
        self.value = tf.Variable(0.0, dtype=tf.float64, name='SE_value')
        self.n_nodes = n_nodes
        self.n_edges = n_edges

    def update_state(self, y_true, y_pred, sample_weight=None):
        l = loss_mae(y_pred=y_pred[:, self.lb:self.ub], y_true=y_true[:, self.lb:self.ub])
        self.value.assign(l)

    def get_config(self):
        return {
            'n_nodes': self.n_nodes,
            'n_edges': self.n_edges,
            'name': self.name
        }


class MetricElementwiseAE(CustomMetric):
    def __init__(self, n_inputs, name='Elementwise_AE', **kwargs):
        super().__init__(name=name)
        self.n_inputs = n_inputs
        self.value = tf.Variable(tf.zeros(n_inputs, dtype=tf.float64), dtype=tf.float64, name='SE_value')

    def update_state(self, y_true, y_pred, sample_weight=None):
        l = tf.reduce_mean(tf.math.abs(y_true - y_pred), axis=0)
        self.value.assign(l)

    def get_config(self):
        return {
            'n_inputs': self.n_inputs,
            'name': self.name
        }


#%% load DNN from save file
def load_DNN(model_file, weight_file=None):
    # loads a DNN from a save file including the custom objects defined above
    model = keras.models.load_model(model_file,
                                    custom_objects={'MyScalingLayer': MyScalingLayer,
                                                    'custom_add': custom_add,
                                                    'get_demands': get_demands,
                                                    'LossSE': LossSE,
                                                    'LossMeasurement': LossMeasurement,
                                                    'LossPriorDemands': LossPriorDemands,
                                                    'LossCombined': LossCombined,
                                                    'LossPenalty': LossPenalty,
                                                    'LossWeightedMSE': LossWeightedMSE,
                                                    'LossWeightedCombinedLoss': LossWeightedCombinedLoss,
                                                    'LossPower': LossPower,

                                                    'MetricSE': MetricSE,
                                                    'MetricMeasurement': MetricMeasurement,
                                                    'MetricPriorDemands': MetricPriorDemands,
                                                    'MetricMAPE_T': MetricMAPE_T,
                                                    'MetricMAPE_mf': MetricMAPE_mf,
                                                    'MetricMAPE_p': MetricMAPE_p,
                                                    'MetricMAPE_Tend': MetricMAPE_Tend,
                                                    'MetricMAE_T': MetricMAE_T,
                                                    'MetricMAE_mf': MetricMAE_mf,
                                                    'MetricMAE_p': MetricMAE_p,
                                                    'MetricMAE_Tend': MetricMAE_Tend,
                                                    'MetricElementwiseAE': MetricElementwiseAE
                                                    })
    # if weight file is specified, load weights as well.
    if weight_file is not None:
        model.load_weights(weight_file).expect_partial()
    return model


class WrappedModel(object):
    '''
    This class builds a wrapper around a DNN with multiple inputs / outputs.
    If input-masks are provided, the input values for these inputs are fixed.
    If an output-selector is choosen, only these outputs of the model will be returned.

    '''
    def __init__(self, inner_model, input_masks=None, output_selector=None):
        self.inner_model = inner_model
        self.n_inputs = len(inner_model.inputs)
        self.mask_names = {i: name for i, name in enumerate(inner_model.input_names)}
        if input_masks is None:
            input_masks = dict()
        self.masks = {name: input_masks.get(name, tf.zeros(shape=(1, shape[1]), dtype=inner_model.dtype)) \
                                            for name, shape in zip(inner_model.input_names, inner_model.input_shape)}
        self.passed_inputs = [i for i, val in enumerate(self.mask_names.values()) if val not in input_masks.keys()]
        if output_selector is None:
            self.output_selector = [i for i, _ in enumerate(inner_model.outputs)]
        else:
            self.output_selector = output_selector

    @tf.function
    def __call__(self, inputs):
        # return self.call([inputs])
        if type(inputs) is list:
            return self.call(inputs)
        else:
            return self.call([inputs])

    def call(self, inputs):
        batch = inputs[0].get_shape()[0]
        # map inputs to input-positions and map input-mask values to masked inputs
        inner_input = []
        j = 0
        for i in range(self.n_inputs):
            if i in self.passed_inputs:
                inner_input.append(inputs[j])
                j += 1
            else:
                inner_input.append(tf.repeat(self.masks[self.mask_names[i]], batch, axis=0))

        # input = [input if i == self.pos_input
        #          else tf.repeat(mv, batch, axis=0) for i, mv in enumerate(self.input_mask)]

        inner_result = self.inner_model(inner_input)
        # return inner model outputs specified by output_selector
        if type(self.output_selector) is list:
            return [inner_result[o] for o in self.output_selector]
        else:
            return inner_result[self.output_selector]

class PartialLoss(object):
    # can't inherent from keras.losses.Loss, because latter would average the loss over all dimensions;
    '''
    This class implements a loss function that applied individual to each state dimension.
    '''
    def __init__(self, SE, loss='MSE', name='PartialLoss', **kwargs):
        """
        Parameters
        ----------
        SE : StateEstimator
            State estimator object.
        loss : str or keras.losses.Loss, optional
            Loss function to be applied to each state dimension. The default is 'MSE'.
        name : str, optional
            Name of the loss function. The default is 'PartialLoss'.
        """
        self.name = name
        # define loss function:
        if type(loss) is str:
            if loss == 'MSE':
                self.loss = keras.losses.MeanSquaredError()
            elif loss == 'RMSE':
                self.loss = lambda y_true, y_pred: tf.sqrt(keras.losses.MeanSquaredError()(y_true, y_pred))
            elif loss == 'MAE':
                self.loss = keras.losses.MeanAbsoluteError()
            elif loss == 'MAPE':
                self.loss = keras.losses.MeanAbsolutePercentageError()
            else:
                raise ValueError(f'Unknown loss function {loss}')
        elif type(loss) is keras.losses.Loss:
            self.loss = loss
        else:
            raise ValueError(f'Unknown loss function {loss}')
        # define partitions:
        self.n_nodes = SE.n_nodes
        self.n_edges = SE.n_edges
        self.T_partition = range(0, self.n_nodes)
        self.mf_partition = range(self.n_nodes, self.n_nodes + self.n_edges)
        self.p_partition = range(self.n_nodes + self.n_edges, 2 * self.n_nodes + self.n_edges)
        self.Tend_partition = range(2 * self.n_nodes + self.n_edges, 2 * self.n_nodes + 2 * self.n_edges)

    def call(self, y_true, y_pred):
        # split y_true and y_pred into partitions:
        y_true_T = tf.gather(y_true, self.T_partition, axis=-1)
        y_true_mf = tf.gather(y_true, self.mf_partition, axis=-1)
        y_true_p = tf.gather(y_true, self.p_partition, axis=-1)
        y_true_Tend = tf.gather(y_true, self.Tend_partition, axis=-1)
        y_pred_T = tf.gather(y_pred, self.T_partition, axis=-1)
        y_pred_mf = tf.gather(y_pred, self.mf_partition, axis=-1)
        y_pred_p = tf.gather(y_pred, self.p_partition, axis=-1)
        y_pred_Tend = tf.gather(y_pred, self.Tend_partition, axis=-1)
        # calculate losses:
        loss_T = self.loss(y_true_T, y_pred_T)
        loss_mf = self.loss(y_true_mf, y_pred_mf)
        loss_p = self.loss(y_true_p, y_pred_p)
        loss_Tend = self.loss(y_true_Tend, y_pred_Tend)
        # return losses:
        return [loss_T, loss_mf, loss_p, loss_Tend]

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

#%% plot functions:
def plot_history(history, save_param=None):
    if save_param is not None:
        save_param = f'{save_param}_'
    else:
        save_param = ''
    epochs = history.epoch
    keys = list(history.history.keys())
    fig_train, ax_train = plt.subplots()
    fig_val, ax_val = plt.subplots()

    for key in keys:
        values = history.history[key]
        label = ''
        if 'loss' in key:
            label = 'loss'
        if 'MAPE' in key:
            label += 'percentage error '
        if 'mean_absolute_percentage_error' in key:
            label += 'percentage error'
        if 'MSE' in key:
            label += 'mean squared error '
        if '_T' == key[-2:]:
            label += 'T'
        if '_mf' == key[-3:]:
            label += 'mf'
        if '_p' == key[-2:]:
            label += 'p'
        if '_Tend' == key[-5:]:
            label += 'Tend'
        if label != '':
            if 'val' in key:
                ax_val.plot(epochs, values, label=label)
            else:
                ax_train.plot(epochs, values, label=label)

    ax_train.set_xlabel('epochs')
    ax_train.set_ylabel('loss')
    ax_train.set_title('training data')
    fig_train.legend()
    fig_train.savefig(f'temp/{save_param}train_loss.svg')

    ax_val.set_xlabel('epochs')
    ax_val.set_ylabel('loss')
    ax_val.set_title('validation data')
    fig_val.legend()
    fig_val.savefig(f'temp/{save_param}val_loss.svg')
    plt.show(block=False)


#%% junkyard:
class LossSEold(keras.losses.Loss):
    def __init__(self, SE=None, SE_config=None, name='SE_loss', run_eager=False):
        super().__init__(name=name)
        # duplicate SE-object; this allows to retrace internal functions with different parameters, i.e. without demands
        if SE is not None:
            config = SE.get_config()
        else:
            config = SE_config
        config['use_demands'] = False
        self.SE_config = config
        self.SE = StateEquations(**config)
        self.n_nodes = self.SE.n_nodes
        self.n_edges = self.SE.n_edges
        self.class_name = 'LossSE'
        self.run_eager = run_eager

    @tf.function
    def call(self, y_true, y_pred):
        """ y_true -> void, y_pred: predicted state, returns left hand side of  state equations """
        [T, mf, p, T_end] = tf.split(y_pred, num_or_size_splits=[self.n_nodes, self.n_edges,
                                                                 self.n_nodes, self.n_edges], axis=-1)
        if self.run_eager:
            lse = [loss_SE(self.SE, T[i:i+1,:], mf[i:i+1,:], p[i:i+1,:], T_end[i:i+1,:]) for i in range(T.get_shape()[0])]
        else:
            lse = tf.map_fn(lambda args: loss_SE(self.SE, tf.expand_dims(args[0], axis=0), tf.expand_dims(args[1], axis=0),
                                                 tf.expand_dims(args[2], axis=0), tf.expand_dims(args[3], axis=0)),
                            elems=(T, mf, p, T_end), fn_output_signature=tf.float64)
        return tf.reduce_mean(lse)

    def get_config(self):
        return {'SE_config': self.SE_config,
                'name': self.name}