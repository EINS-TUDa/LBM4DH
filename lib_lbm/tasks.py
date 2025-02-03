import time
import os
import warnings
from .utility.utility import Task, DNNTask, get_scenario_params, create_scenario
from .utility.DataClasses import DnnData
import numpy as np
import datetime
import tensorflow as tf
from tensorflow import keras
import lib_lbm.importance_sampling as IS
import lib_lbm.se_NN_lib as NN
from lib_lbm.classic_solver import SE_solver

solve_SE = SE_solver()

class DummyTask(Task):
    """ dummy task for debugging """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def execute(self):
        print(self.scenario_name)
        return dict()

    def report(self):
        time_str = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
        res_str = f'numerical results for testcase {self.scenario_name} calculated at {time_str} \n\n'
        super().report(res_str)

class ResetOutputFile(Task):
    """ Clean up the output file """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def execute(self):
        with open(self.output_file, 'w') as f:
            f.write('')
        return {'output_file': self.output_file}

    def report(self):
        time_str = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
        res_str = f'numerical results for testcase {self.scenario_name} calculated at {time_str} \n\n'
        super().report(res_str)


class SetupScenario(Task):
    """ Setup for all experiments """
    def __init__(self, scenario, create_new=False, pp_pos=None, grid_dim=None, output_file=None, **kwargs):
        """
        :param scenario:    name of the scenario
        :param create_new:  boolean or 'auto': whether to create a new grid file for this scenario
        :param pp_pos:      required if create_new = True;  positions of power plants in the grid
        :param grid_dim:    required if create_new = True;  distance between two active edges
        :param output_file: output_file: output print file for the current experiment
        :param kwargs:      passed to task
        """
        self.scenario_name = scenario
        self.new = create_new if type(create_new) is bool else not self._scenario_exists(scenario)
        self.pp_pos = pp_pos
        self.grid_dim = grid_dim
        self._output_file_arg = output_file  # store the value passed on initialisation
        if self.new:
            if pp_pos is None or grid_dim is None:
                raise Exception('In order to create a new grid file please specify "grid_dim" and "pp_pos"')
        super().__init__(**kwargs)

    def _scenario_exists(self, scenario_name):
        return os.path.exists(f'../../grids/{scenario_name}.json')

    def execute(self):
        if self.new:
            create_scenario(self.scenario_name, pp_pos=self.pp_pos, grid_dim=self.grid_dim)
        scen_params = get_scenario_params(self.scenario_name, output_file=self._output_file_arg)
        super()._set_scenario_attributes(scen_params)
        return {'scenario_params': scen_params}

    def report(self):
        if self.new:
            res_str = f'Created new grid file with grid_dim {self.grid_dim} and heat supplies at position ' \
                      f'{self.pp_pos} \n\n'
        else:
            res_str = f'Loaded grid file named {self.scenario_name} \n\n'
        super().report(res_str)


class CompareSampleTimes(Task):
    """ compare the run times for the classic DC-NR algorithm and IS sampling """
    def __init__(self, n_samples, compute_NR=True, compute_IS=True, compensate_esr=False, **kwargs):
        """
        :param n_samples:       Number of samples in comparison
        :param compute_NR:      boolean: whether to execute DC-NR sampling
        :param compute_IS:      boolean: whether to execute IS sampling
        :param compensate_esr:  boolean: if true:
                                    increase sample size of IS sampling to compensate for reduced effective sample rate
        """
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.compensate_esr = compensate_esr
        self.compute_IS = compute_IS
        self.compute_NR = compute_NR
        self.is_surplus = 0.1   # initial setting: use a surplus of 10% during IS sampling

        # values calculated during execution:
        self.n_eff = None           # number of effective samples drawn in IS
        self.eff_sr = None          # effective sample rate
        self.setup_time_NR = None   # time to setup NR sampling (i.e. first run setting up TF functions)
        self.setup_time_IS = None   # time to setup IS sampling including calculating proxy distribution
        self.run_time_NR = None     # time to draw settings['n_samples'] using NR sampling
        self.run_time_IS = None     # time to draw settings['n_samples'] using IS sampling

    def execute(self):
        def NR_sampling(n_samples):
            '''  ------------  classical sampling:   ------------   '''
            # setup:
            start = datetime.datetime.now()
            IS.setup_d_sampling(self.d_dist, self.T_dist, self.SE, self.cycles)
            start_sampling = datetime.datetime.now()
            IS.timing_d_sampling(n_samples, self.SE, self.cycles, self.d_dist, self.T_dist, verbose=self.verbose)
            end = datetime.datetime.now()
            return start, start_sampling, end

        def _IS_sampling(mf_dist, n_samples):
            '''  --------------- IS sampling ---------------------  '''
            start = datetime.datetime.now()
            n_eff = IS.timing_mf_sampling(n_samples, self.SE, self.cycles, self.grid, self.d_dist, self.T_dist, mf_dist,
                                          verbose=self.verbose)
            end = datetime.datetime.now()
            return n_eff, start, end

        def IS_sampling(n_samples):
            # setup sampling:
            start_IS_setup = datetime.datetime.now()
            mf_dist = IS.setup_mf_sampling(self.d_dist, self.T_dist, self.SE, self.cycles)
            end_IS_setup = datetime.datetime.now()

            # run sampling:
            if self.compensate_esr:
                n_eff = 0
                while n_eff < n_samples:
                    # perform IS sampling, increase is_surplus until the effective sample number is equal to the desired
                    n_is_samples = int(n_samples * (1 + self.is_surplus))
                    n_eff, start_sampling_IS, end_sampling_Is = _IS_sampling(mf_dist, n_is_samples)
                    self.is_surplus += 0.1
                self.is_surplus -= 0.1
            else:
                n_is_samples = n_samples
                n_eff, start_sampling_IS, end_sampling_Is = _IS_sampling(mf_dist, n_samples)
            return n_eff, start_IS_setup, end_IS_setup, start_sampling_IS, end_sampling_Is, n_is_samples

        # run NR_sampling:
        if self.compute_NR:
            start_NR, start_sampling_NR, end_NR = NR_sampling(self.n_samples)
        else:
            start_NR = start_sampling_NR = end_NR = datetime.datetime.now()

        # setup IS sampling:
        if self.compute_IS:
            n_eff, start_IS_setup, end_IS_setup, start_sampling_IS, end_sampling_Is, n_is_samples = IS_sampling(self.n_samples)
        else:
            start_IS_setup = end_IS_setup = start_sampling_IS = end_sampling_Is = datetime.datetime.now()
            n_is_samples = 0
            n_eff = 0

        # save results:
        self.n_eff = n_eff  # effective number of samples and effective sample rate
        self.eff_sr = n_eff / n_is_samples
        self.setup_time_NR = start_sampling_NR - start_NR
        self.setup_time_IS = end_IS_setup - start_IS_setup
        self.run_time_NR = end_NR - start_sampling_NR
        self.run_time_IS = end_sampling_Is - start_sampling_IS
        return dict()

    def report(self):
        res_str = f'################################ results Importance sampling ################################\n' \
                  f'\n' \
                  f'                    |            NR sampling            |              IS sampling           \n' \
                  f'---------------------------------------------------------------------------------------------\n' \
                  f'  eff. sample rate  |                1.0                |                {self.eff_sr * 100:2.2f}%              \n' \
                  f'  num eff. samples  |             {self.n_samples:6}                |             {int(self.n_eff):6}                 \n' \
                  f'---------------------------------------------------------------------------------------------\n' \
                  f'  setup times       |            {self.setup_time_NR}         |            {self.setup_time_IS}          \n' \
                  f'  run times         |            {self.run_time_NR}         |            {self.run_time_IS}          \n' \
                  f'  time per sample   |            {(self.setup_time_NR + self.run_time_NR) / self.n_samples}         |' \
                  f'            {(self.setup_time_IS + self.run_time_IS) / self.n_eff}          \n' \
                  f'\n' \
                  f' Gross time gain: {(self.setup_time_NR + self.run_time_NR) / (self.setup_time_IS + self.run_time_IS):4f}' \
                  f' Net time gain:   {self.run_time_NR / self.run_time_IS:4f}\n\n'

        super().report(res_str)


class GenerateTrainingSamples(Task):
    """ Generates training samples using the IS algorithm """
    def __init__(self, n_samples, include_slack, n_per_file=2_000, ignore_existing=False, overwrite=False, **kwargs):
        """
        :param n_samples: number of samples to draw
        :param include_slack: if True, the power of the slack power plant is included in the data
        :param n_per_file: number of samples per file
        :param ignore_existing: if True n_samples are drawn and saved, even if there are already existing samples
        :param overwrite: if True, existing samples are overwritten - only works if ignore_existing is True
        """

        super().__init__(**kwargs)
        if overwrite and not ignore_existing:
            raise Exception('overwrite can only be True if ignore_existing is True')
        mum_existed = self._number_of_existing_samples()
        if overwrite:
            self.num_existed = 0
            self.n_samples = n_samples
            for i in range(mum_existed // n_per_file):
                os.remove(self.training_data_file(i))
        else:
            self.num_existed = mum_existed
            self.n_samples = n_samples if ignore_existing else max(n_samples - mum_existed, 0)
        self.slack = include_slack
        self.n_per_file = n_per_file

        # time keeping variables:
        self.start_setup = None
        self.start_sampling = None
        self.end_sampling = None
        self.start_setup_wall = None
        self.start_sampling_wall = None
        self.end_sampling_wall = None

    def _number_of_existing_samples(self):
        files = 0
        samples = 0
        while os.path.exists(self.training_data_file(files)):
            with open(self.training_data_file(files), 'r') as f:
                for line in f:
                    samples += 1  # count lines
            files += 1
        return samples

    def execute(self):
        # check if there are any samples to draw:
        if not self.n_samples > 0:
            # set all times to now:
            self.start_setup = self.start_sampling = self.end_sampling = time.process_time()
            self.start_setup_wall = self.end_sampling_wall = datetime.datetime.now()
            if self.verbose:
                print(f'No new samples to draw, skipping sampling step')
            return dict()

        # setup results directory:
        if not os.path.exists(self._training_data_path):
            os.makedirs(self._training_data_path)

        # setup IS sampling
        # file spec is corrected to fill up the last file with samples and start a new one, once n_per_file is reached
        filespec = lambda i: self.training_data_file((i+self.num_existed) // self.n_per_file)
        sampler = IS.ImportanceSampler(self.d_dist, self.T_dist, self.SE, self.cycles, self.grid)
        self.start_setup = time.process_time()
        self.start_setup_wall = datetime.datetime.now()
        sampler.setup()

        # draw samples
        self.start_sampling = time.process_time()
        self.start_sampling_wall = datetime.datetime.now()
        sampler(n_samples=self.n_samples, verbose=self.verbose, include_slack=self.slack)
        self.end_sampling = time.process_time()
        self.end_sampling_wall = datetime.datetime.now()
        return dict()

    def report(self):
        res_str = f'{"_"*120}\n' \
                  f'generation of samples type: {self.name} \n' \
                  f'{"_"*120}\n' \
                  f'Successfully generated {self.n_samples} ' \
                  f'samples and saved them under {self.training_data_file("XXX")} \n' \
                  f'cpu sampling time: {self.end_sampling - self.start_sampling} \n' \
                  f'wall total time: {self.end_sampling_wall - self.start_setup_wall} \n' \
                  f'cpu total time: {self.end_sampling - self.start_setup} \n ' \
                  f'{"="*120}\n\n'

        super().report(res_str)


class BuildDNN(DNNTask):
    """ Setup the DNN to be trained """
    def __init__(self, layers, add_demands=True, include_slack_output=False, include_slack_input=False,
                 n_normalise_samples=1000, **kwargs):
        """
        :param layers:                  number of nodes per trainable hidden layer (+ 1 layer with size equals output)
        :param add_demands:             boolean: whether to add calculated demands as second output
        :param include_slack_output:    boolean: whether output vector contains a value for the slack power plant
        :param include_slack_input:     boolean: whether input vector contains a value for the slack power plant
        :param n_normalise_samples:     int: number of samples used to calculate normalisaiton factors
        :param kwargs:                  passed to DNNTask
        """
        self.layers_spec = layers
        self.add_demands = add_demands
        self.add_slack_output = include_slack_output
        self.add_slack_input = include_slack_input
        self.n_normalise_samples = n_normalise_samples
        super().__init__(**kwargs)

    def execute(self):
        SE = self.SE
        # if no data handler is provided
        if self.Data is None:
            self.Data = DnnData(self.training_data_file(''), self.SE, n_train=0,
                                d_dist=self.d_dist, T_dist=self.T_dist,
                                include_slack_input=self.add_slack_input, include_slack_output=self.add_slack_output)

        dem_index_list = SE.dem_order
        heat_index_list = SE.pp_order

        if self.n_normalise_samples > 0:
            x_input, y_input = self.Data.get_data(self.n_normalise_samples, section='train')
            norm_on_data = True
            model = NN.build_DNN(SE, dem_index_list, heat_index_list, *x_input, y_input[0],
                                 normalise_output_on_data=norm_on_data,
                                 d_prior_dist=self.d_dist, T_prior_dist=self.T_dist,
                                 layer_spec=self.layers_spec,
                                 add_demands_output=True, add_slack_output=self.add_slack_output, cycles=self.cycles)
            model.save(self.model_weight_file('init'))
            if self.DNN is None:
                self.DNN = model
            return {'DataHandler': self.Data, 'DNN': model}
        else:
            raise Exception('Number of normalisation samples can not be 0')
            '''
            Idea: draw random input samples for normalisation 
                  use 1st order Taylor expansion to normalise output data (pass norm_on_data=False to NN.buio_DNN(...)) 
            
            This doe's not work on larger grids (>= 4 active edges) -> workflow sketched out as: 
            '''
            n_pseudo_samples = 1_000
            dg = self.Data.get_data_generator(n_pseudo_samples, n_pseudo_samples)
            x_input, y_input = dg[0]
            x_input = [tf.reshape(x, (n_pseudo_samples, -1)) for x in x_input]
            # data generator items do never include a slack input, so we have to add it here:
            if self.add_slack_input:
                tf.pad(x_input[1], tf.constant([[0, 0], [0, 1]]), 'Symmetric')
            y_input = [tf.reshape(y, (n_pseudo_samples, -1)) for y in y_input]
            norm_on_data = False


            model = NN.build_DNN(SE, dem_index_list, heat_index_list, *x_input, y_input[0],
                                 normalise_output_on_data=norm_on_data,
                                 d_prior_dist=self.d_dist, T_prior_dist=self.T_dist,
                                 layer_spec=self.layers_spec,
                                 add_demands_output=True, add_slack_output=self.add_slack_output, cycles=self.cycles)

            model.save(self.model_weight_file('init'))
            if self.DNN is None:
                self.DNN = model
            return {'DataHandler': self.Data, 'DNN': model}

    def report(self):
        str_list = []
        self.DNN.summary(line_length=120, print_fn=lambda s: str_list.append(s))
        model_str = '\n'.join(str_list)
        res_str = f'Create new DNN with the following structure: \n {model_str} \n' \
                  f'Save model structure under "{self.model_weight_file("init")}"\n\n'
        super().report(res_str)


class TrainDNN(DNNTask):
    """ Trains the DNN with a WMSE or PAL """
    def __init__(self, training_settings, verbose='auto', n_test_samples=500, **kwargs):
        """
        :param training_settings:   training parameters
        :param verbose:             verbosity
        :param n_test_samples:      number of samples used for model evaluation
        :param kwargs:              passed to DNNTask
        """
        self.start_epoch = training_settings.get('start_epoch', 0)
        self.settings = training_settings
        self.n_eval_samples = n_test_samples
        self.verbose = verbose
        self.history = None
        self.training_time = None
        super().__init__(**kwargs)

        # load weights if pretrained model is used
        if self.start_epoch != 0:
            self.DNN.load_weights(self.model_weight_file(self.start_epoch))
        if type(self.start_epoch) is not int:
            self.start_epoch = 0

        # setup optimiser:
        try:
            self.opt = keras.optimizers.get({'class_name': self.settings['optimizer'],
                                             'config': self.settings['opt_config']})
        except KeyError:
            self.opt = None
        # setup callbacks:
        self.setup_callbacks(training_settings)

    def setup_callbacks(self, training_settings):
        """
        supported callbacks are:
        'save model':       int or True: saves the model weights every 'int' epochs or 100 times if true is passed
        'early stopping':   early stopping callback restoring the best weights after training
        """
        self.callbacks = []
        # model saving in regular intervals
        if training_settings.get('save model', False):
            checkpoint_path = self.model_weight_file()
            # save model every 1% of the training process or every x epochs if x is specified in settings
            epochs_save_freq = self.settings['save model'] if type(self.settings['save model']) is int \
                else int(self.settings['epochs'] / 100)
            # convert save frequency to number of batches
            save_freq = int(
                epochs_save_freq * np.ceil(self.settings['n_train'] / self.settings['batch_size']))
            self.callbacks.append(keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path + '{epoch:04d}', save_freq=save_freq, save_weights_only=True))
        if training_settings.get('early stopping', False):
            if self.settings['loss'] == 'PAL':
                monitor = 'loss'
            else:
                monitor = 'val_loss'
            self.callbacks.append(keras.callbacks.EarlyStopping(
                monitor=monitor, patience=training_settings['early stopping'], verbose=1, restore_best_weights=True))

    def train_L2(self, training_data, validation_data):
        x_t = tf.data.Dataset.from_tensor_slices(tuple(training_data[0]))
        y_t = tf.data.Dataset.from_tensor_slices(tuple(training_data[1]))
        data_train = tf.data.Dataset.zip((x_t, y_t))
        x_v = tf.data.Dataset.from_tensor_slices(tuple(validation_data[0]))
        y_v = tf.data.Dataset.from_tensor_slices(tuple(validation_data[1]))
        data_val = tf.data.Dataset.zip((x_v, y_v))

        t_start = datetime.datetime.now()
        history = self.DNN.fit(data_train.batch(self.settings['batch_size']),
                               validation_data=data_val.batch(self.settings['batch_size']),
                               initial_epoch=self.start_epoch,
                               epochs=self.settings['epochs'] + self.start_epoch,
                               callbacks=self.callbacks,
                               verbose=self.verbose
                               )
        t_end = datetime.datetime.now()
        return history, t_end - t_start

    def train_PA(self, DG):
        data = tf.data.Dataset.from_generator(
            generator=DG,
            output_signature=(
                (tf.TensorSpec(shape=DG[0][0][0].shape, dtype=DG[0][0][0].dtype),
                 tf.TensorSpec(shape=DG[0][0][1].shape, dtype=DG[0][0][1].dtype),
                 tf.TensorSpec(shape=DG[0][0][2].shape, dtype=DG[0][0][2].dtype),
                 tf.TensorSpec(shape=DG[0][0][3].shape, dtype=DG[0][0][3].dtype)),
                (tf.TensorSpec(shape=DG[0][1][0].shape, dtype=DG[0][1][0].dtype),
                 tf.TensorSpec(shape=DG[0][1][1].shape, dtype=DG[0][1][1].dtype))
            ))

        # whatever this code does, but it is what is written in the warning that appears without this code
        dataset_options = tf.data.Options()
        dataset_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        data = data.with_options(dataset_options)

        t_start = datetime.datetime.now()
        history = self.DNN.fit(data,
                               batch_size=self.settings['batch_size'],
                               initial_epoch=self.start_epoch,
                               epochs=self.settings['epochs'] + self.start_epoch,
                               callbacks=self.callbacks,
                               use_multiprocessing=True,
                               verbose=self.verbose
                               )
        t_end = datetime.datetime.now()

        return history, t_end - t_start

    def execute(self):
        if self.settings['loss'] == 'MSE':
            # setup loss function for DNN
            state_loss = NN.LossWeightedMSE(self.SE.n_nodes, self.SE.n_edges, lambda_mf=500)
            power_loss = keras.losses.MeanSquaredError() if self.settings['power MSE'] else None
            loss = [state_loss, power_loss]
            self.DNN.compile(loss=loss, optimizer=self.opt)

            # load training and validation data:
            training_data = self.Data.get_data(self.settings['n_train'], section='train')
            validation_data = self.Data.get_data(self.settings['n_val'], section='validation')

            history, training_time = self.train_L2(training_data, validation_data)

        elif self.settings['loss'] == 'PAL':
            # setup loss function for DNN
            state_loss = NN.LossSE(self.SE)
            assert self.DNN.output_shape == self.Data.output_shape, \
                f'DNN output shape {self.DNN.output_shape} 'f'does not match data output shape {self.Data.output_shape}'
            if self.Data.include_slack_output:
                warnings.warn('Slack output is included in the loss function, but not in the true data. It will be '
                              'ignored in the loss function.')
            power_loss = NN.LossPower(y_shape=self.DNN.output_shape[-1], mask_slack=self.Data.include_slack_output)
            loss = [state_loss, power_loss]
            # loss = [state_loss, None]
            self.DNN.compile(loss=loss, optimizer=self.opt)

            # load training data - generate new samples each epoch, no need for validation data
            data_generator = self.Data.get_data_generator(self.settings['n_train'], self.settings['batch_size'])
            history, training_time = self.train_PA(data_generator)
        else:
            raise Exception('Unknown loss function')

        self.history = history
        self.training_time = training_time
        self.DNN.save_weights(self.model_weight_file(f'final_{self.settings["loss"]}'))
        self.DNN.save(self.model_weight_file(f'final_{self.settings["loss"]}'))
        return dict()

    def evaluate_performance(self):
        test_data = self.Data.get_data(self.n_eval_samples, section='test')
        prediction = self.DNN(test_data[0])

        MSE_state = keras.losses.MeanSquaredError()(test_data[1][0], prediction[0])
        MSE_power = keras.losses.MeanSquaredError()(test_data[1][1], prediction[1])
        MAE_state = keras.losses.MeanAbsoluteError()(test_data[1][0], prediction[0])
        MAE_power = keras.losses.MeanAbsoluteError()(test_data[1][1], prediction[1])
        # normalise physical loss by number of state variables (2 per node and edge)
        PAL = NN.LossSE(self.SE)(tf.zeros_like(prediction[0]), prediction[0])/(2 * (self.SE.n_nodes + self.SE.n_edges))

        # partial losses for each state dimension::
        P_MAE = NN.PartialLoss(self.SE, loss='MAE', name='SMAE')(test_data[1][0], prediction[0])
        P_RMSE = NN.PartialLoss(self.SE, loss='RMSE', name='SRMSE')(test_data[1][0], prediction[0])
        P_MAPE = NN.PartialLoss(self.SE, loss='MAPE', name='SMAPE')(test_data[1][0], prediction[0])

        print(f'final PAL: {PAL}')
        print(f'final MAE_Power: {MAE_power}')

        return MSE_state, MSE_power, MAE_state, MAE_power, PAL, P_MAE, P_RMSE, P_MAPE

    def report(self):
        MSE_state, MSE_power, MAE_state, MAE_power, PAL, P_MAE, P_RMSE, P_MAPE = self.evaluate_performance()
        settings = self.settings
        if settings['loss'] == 'MSE':
            dem_loss_str = f"    - loss function power values: {'MSE' if settings['power MSE'] else 'None'} \n"
        else:
            dem_loss_str = ''

        res_str = f"trained DNN with loss-function {settings['loss']} in {self.training_time} \n" \
                  f"Settings: \n" \
                  f"    - loss function grid state: {settings['loss']} \n" \
                  f"    - start epoch: {self.start_epoch} \n" \
                  f"{dem_loss_str}" \
                  f"    - epochs: {settings['epochs']} \n" \
                  f"    - n_train: {settings['n_train']} \n" \
                  f"    - batch_size: {settings['batch_size']} \n" \
                  f"    - optimiser: {self.settings['optimizer']}\n" \
                  f"        - config: {self.settings['opt_config']} \n\n" \
                  f"Performance of trained DNN: " \
                  f"state prediction: \n" \
                  f"    - RMSE:         {tf.reduce_mean(tf.math.sqrt(MSE_state))} \n" \
                  f"    - MAE:          {tf.reduce_mean(MAE_state)} \n" \
                  f"    - Phys. loss:   {tf.reduce_mean(PAL)} \n" \
                  f"    - partial losses: \n" \
                  f"        - MAE  T:   {tf.reduce_mean(P_MAE[0])} \n" \
                  f"        - RMSE T:   {tf.reduce_mean(P_RMSE[0])} \n" \
                  f"        - MAPE T:   {tf.reduce_mean(P_MAPE[0])} \n" \
                  f"        - MAE  mf:  {tf.reduce_mean(P_MAE[1])} \n" \
                  f"        - RMSE mf:  {tf.reduce_mean(P_RMSE[1])} \n" \
                  f"        - MAPE mf:  {tf.reduce_mean(P_MAPE[1])} \n" \
                  f"        - MAE  p:   {tf.reduce_mean(P_MAE[2])} \n" \
                  f"        - RMSE p:   {tf.reduce_mean(P_RMSE[2])} \n" \
                  f"        - MAPE p:   {tf.reduce_mean(P_MAPE[2])} \n" \
                  f"        - MAE Tend: {tf.reduce_mean(P_MAE[3])} \n" \
                  f"        - RMSE Tend: {tf.reduce_mean(P_RMSE[3])} \n" \
                  f"        - MAPE Tend: {tf.reduce_mean(P_MAPE[3])} \n" \
                  f"power deviation: \n" \
                  f"    - RMSE:         {tf.reduce_mean(tf.math.sqrt(MSE_power))} \n" \
                  f"    - MAE:          {tf.reduce_mean(MAE_power)} \n"

        super().report(res_str)


class TrainDNN(DNNTask):
    """ Trains the DNN with a WMSE or PAL """
    def __init__(self, training_settings, verbose='auto', n_test_samples=500, **kwargs):
        """
        :param training_settings:   training parameters
        :param verbose:             verbosity
        :param n_test_samples:      number of samples used for model evaluation
        :param kwargs:              passed to DNNTask
        """
        self.start_epoch = training_settings.get('start_epoch', 0)
        self.settings = training_settings
        self.n_eval_samples = n_test_samples
        self.verbose = verbose
        self.history = None
        self.training_time = None
        super().__init__(**kwargs)

        # load weights if pretrained model is used
        if self.start_epoch != 0:
            self.DNN.load_weights(self.model_weight_file(self.start_epoch))
        if type(self.start_epoch) is not int:
            self.start_epoch = 0

        # setup optimiser:
        try:
            self.opt = keras.optimizers.get({'class_name': self.settings['optimizer'],
                                             'config': self.settings['opt_config']})
        except KeyError:
            self.opt = None
        # setup callbacks:
        self.setup_callbacks(training_settings)

    def setup_callbacks(self, training_settings):
        """
        supported callbacks are:
        'save model':       int or True: saves the model weights every 'int' epochs or 100 times if true is passed
        'early stopping':   early stopping callback restoring the best weights after training
        """
        self.callbacks = []
        # model saving in regular intervals
        if training_settings.get('save model', False):
            checkpoint_path = self.model_weight_file()
            # save model every 1% of the training process or every x epochs if x is specified in settings
            epochs_save_freq = self.settings['save model'] if type(self.settings['save model']) is int \
                else int(self.settings['epochs'] / 100)
            # convert save frequency to number of batches
            save_freq = int(
                epochs_save_freq * np.ceil(self.settings['n_train'] / self.settings['batch_size']))
            self.callbacks.append(keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path + '{epoch:04d}', save_freq=save_freq, save_weights_only=True))
        if training_settings.get('early stopping', False):
            if self.settings['loss'] == 'PAL':
                monitor = 'loss'
            else:
                monitor = 'val_loss'
            self.callbacks.append(keras.callbacks.EarlyStopping(
                monitor=monitor, patience=training_settings['early stopping'], verbose=1, restore_best_weights=True))

    def train_L2(self, training_data, validation_data):
        x_t = tf.data.Dataset.from_tensor_slices(tuple(training_data[0]))
        y_t = tf.data.Dataset.from_tensor_slices(tuple(training_data[1]))
        data_train = tf.data.Dataset.zip((x_t, y_t))
        x_v = tf.data.Dataset.from_tensor_slices(tuple(validation_data[0]))
        y_v = tf.data.Dataset.from_tensor_slices(tuple(validation_data[1]))
        data_val = tf.data.Dataset.zip((x_v, y_v))

        t_start = datetime.datetime.now()
        history = self.DNN.fit(data_train.batch(self.settings['batch_size']),
                               validation_data=data_val.batch(self.settings['batch_size']),
                               initial_epoch=self.start_epoch,
                               epochs=self.settings['epochs'] + self.start_epoch,
                               callbacks=self.callbacks,
                               verbose=self.verbose
                               )
        t_end = datetime.datetime.now()
        return history, t_end - t_start

    def train_PA(self, DG):
        data = tf.data.Dataset.from_generator(
            generator=DG,
            output_signature=(
                (tf.TensorSpec(shape=DG[0][0][0].shape, dtype=DG[0][0][0].dtype),
                 tf.TensorSpec(shape=DG[0][0][1].shape, dtype=DG[0][0][1].dtype),
                 tf.TensorSpec(shape=DG[0][0][2].shape, dtype=DG[0][0][2].dtype),
                 tf.TensorSpec(shape=DG[0][0][3].shape, dtype=DG[0][0][3].dtype)),
                (tf.TensorSpec(shape=DG[0][1][0].shape, dtype=DG[0][1][0].dtype),
                 tf.TensorSpec(shape=DG[0][1][1].shape, dtype=DG[0][1][1].dtype))
            ))

        # whatever this code does, but it is what is written in the warning that appears without this code
        dataset_options = tf.data.Options()
        dataset_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        data = data.with_options(dataset_options)

        t_start = datetime.datetime.now()
        history = self.DNN.fit(data,
                               batch_size=self.settings['batch_size'],
                               initial_epoch=self.start_epoch,
                               epochs=self.settings['epochs'] + self.start_epoch,
                               callbacks=self.callbacks,
                               use_multiprocessing=True,
                               verbose=self.verbose
                               )
        t_end = datetime.datetime.now()

        return history, t_end - t_start

    def execute(self):
        if self.settings['loss'] == 'MSE':
            # setup loss function for DNN
            state_loss = NN.LossWeightedMSE(self.SE.n_nodes, self.SE.n_edges, lambda_mf=500)
            power_loss = keras.losses.MeanSquaredError() if self.settings['power MSE'] else None
            loss = [state_loss, power_loss]
            self.DNN.compile(loss=loss, optimizer=self.opt)

            # load training and validation data:
            training_data = self.Data.get_data(self.settings['n_train'], section='train')
            validation_data = self.Data.get_data(self.settings['n_val'], section='validation')

            history, training_time = self.train_L2(training_data, validation_data)

        elif self.settings['loss'] == 'PAL':
            # setup loss function for DNN
            state_loss = NN.LossSE(self.SE)
            assert self.DNN.output_shape == self.Data.output_shape, \
                f'DNN output shape {self.DNN.output_shape} 'f'does not match data output shape {self.Data.output_shape}'
            if self.Data.include_slack_output:
                warnings.warn('Slack output is included in the loss function, but not in the true data. It will be '
                              'ignored in the loss function.')
            power_loss = NN.LossPower(y_shape=self.DNN.output_shape[-1], mask_slack=self.Data.include_slack_output)
            loss = [state_loss, power_loss]
            # loss = [state_loss, None]
            self.DNN.compile(loss=loss, optimizer=self.opt)

            # load training data - generate new samples each epoch, no need for validation data
            data_generator = self.Data.get_data_generator(self.settings['n_train'], self.settings['batch_size'])
            history, training_time = self.train_PA(data_generator)
        else:
            raise Exception('Unknown loss function')

        self.history = history
        self.training_time = training_time
        self.DNN.save_weights(self.model_weight_file(f'final_{self.settings["loss"]}'))
        self.DNN.save(self.model_weight_file(f'final_{self.settings["loss"]}'))
        return dict()

    def evaluate_performance(self):
        test_data = self.Data.get_data(self.n_eval_samples, section='test')
        prediction = self.DNN(test_data[0])

        MSE_state = keras.losses.MeanSquaredError()(test_data[1][0], prediction[0])
        MSE_power = keras.losses.MeanSquaredError()(test_data[1][1], prediction[1])
        MAE_state = keras.losses.MeanAbsoluteError()(test_data[1][0], prediction[0])
        MAE_power = keras.losses.MeanAbsoluteError()(test_data[1][1], prediction[1])
        # normalise physical loss by number of state variables (2 per node and edge)
        PAL = NN.LossSE(self.SE)(tf.zeros_like(prediction[0]), prediction[0])/(2 * (self.SE.n_nodes + self.SE.n_edges))

        # partial losses for each state dimension::
        P_MAE = NN.PartialLoss(self.SE, loss='MAE', name='SMAE')(test_data[1][0], prediction[0])
        P_RMSE = NN.PartialLoss(self.SE, loss='RMSE', name='SRMSE')(test_data[1][0], prediction[0])
        P_MAPE = NN.PartialLoss(self.SE, loss='MAPE', name='SMAPE')(test_data[1][0], prediction[0])

        print(f'final PAL: {PAL}')
        print(f'final MAE_Power: {MAE_power}')

        return MSE_state, MSE_power, MAE_state, MAE_power, PAL, P_MAE, P_RMSE, P_MAPE

    def report(self):
        MSE_state, MSE_power, MAE_state, MAE_power, PAL, P_MAE, P_RMSE, P_MAPE = self.evaluate_performance()
        settings = self.settings
        if settings['loss'] == 'MSE':
            dem_loss_str = f"    - loss function power values: {'MSE' if settings['power MSE'] else 'None'} \n"
        else:
            dem_loss_str = ''

        res_str = f"trained DNN with loss-function {settings['loss']} in {self.training_time} \n" \
                  f"Settings: \n" \
                  f"    - loss function grid state: {settings['loss']} \n" \
                  f"    - start epoch: {self.start_epoch} \n" \
                  f"{dem_loss_str}" \
                  f"    - epochs: {settings['epochs']} \n" \
                  f"    - n_train: {settings['n_train']} \n" \
                  f"    - batch_size: {settings['batch_size']} \n" \
                  f"    - optimiser: {self.settings['optimizer']}\n" \
                  f"        - config: {self.settings['opt_config']} \n\n" \
                  f"Performance of trained DNN: " \
                  f"state prediction: \n" \
                  f"    - RMSE:         {tf.reduce_mean(tf.math.sqrt(MSE_state))} \n" \
                  f"    - MAE:          {tf.reduce_mean(MAE_state)} \n" \
                  f"    - Phys. loss:   {tf.reduce_mean(PAL)} \n" \
                  f"    - partial losses: \n" \
                  f"        - MAE  T:   {tf.reduce_mean(P_MAE[0])} \n" \
                  f"        - RMSE T:   {tf.reduce_mean(P_RMSE[0])} \n" \
                  f"        - MAPE T:   {tf.reduce_mean(P_MAPE[0])} \n" \
                  f"        - MAE  mf:  {tf.reduce_mean(P_MAE[1])} \n" \
                  f"        - RMSE mf:  {tf.reduce_mean(P_RMSE[1])} \n" \
                  f"        - MAPE mf:  {tf.reduce_mean(P_MAPE[1])} \n" \
                  f"        - MAE  p:   {tf.reduce_mean(P_MAE[2])} \n" \
                  f"        - RMSE p:   {tf.reduce_mean(P_RMSE[2])} \n" \
                  f"        - MAPE p:   {tf.reduce_mean(P_MAPE[2])} \n" \
                  f"        - MAE Tend: {tf.reduce_mean(P_MAE[3])} \n" \
                  f"        - RMSE Tend: {tf.reduce_mean(P_RMSE[3])} \n" \
                  f"        - MAPE Tend: {tf.reduce_mean(P_MAPE[3])} \n" \
                  f"power deviation: \n" \
                  f"    - RMSE:         {tf.reduce_mean(tf.math.sqrt(MSE_power))} \n" \
                  f"    - MAE:          {tf.reduce_mean(MAE_power)} \n"

        super().report(res_str)
