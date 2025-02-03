"""
This file contains the functions for the proposed IS algorithm from the SESAAU paper.

Functions:
    The setup- and timing-functions are used to estimate the time needed for sampling. No IO is performend.
    setup_d_sampling: runs SE-solver for the mean demand and temperature to set the SE to a valid state
    timing_d_sampling: runs SE-solver for a given number of samples to estimate the time needed for sampling
    setup_mf_sampling: sets up the mass flow distribution for IS sampling
    timing_mf_sampling: runs IS-algorithm for a given number of samples to estimate the time needed for sampling

    generate_samples: runs IS-algorithm for a given number of samples and writes the results to a file
"""

import numpy as np
import tensorflow as tf
import warnings
from lib_lbm.utility.state_equations import TemperatureProbagationException
from lib_lbm.classic_solver import my_fixpoint_iteratror, SE_solver, MaximumIterationException
from lib_lbm.utility.DataClasses import ZeroTruncatedMultivariateNormal
solve_SE = SE_solver()
FI = my_fixpoint_iteratror()

def _solve_SE(demands, temperatures, SE, cycles):
    """
    solves the state equations for a given set of demands and temperatures
    :param demands:
    :param temperatures:
    :param SE: State equations object
    :param cycles: loops in the grid
    :return: nothing - SE is modified in place
    """
    def __solve_SE(demands, temperatures, SE, cycles):
        for dem in SE.demands.keys():
            ind = SE.dem_ind[dem]
            d, T = demands[ind], temperatures[ind]
            SE.Q_heat[ind].assign(d)
            SE.set_active_edge_temperature(dem, T)
        # set temperature for heating - last entry in T_vector
        ind = [k for k in SE.heatings.keys()][0]
        SE.set_active_edge_temperature(ind, temperatures[-1])
        solve_SE(SE, cycles, verbose=False)
    try:
        __solve_SE(demands, temperatures, SE, cycles)
    except (MaximumIterationException, TemperatureProbagationException) as e:
        warnings.warn(f'Exception occured: {e} \n reset SE to init state and try solving again')
        SE.set_init_state()
        try:
            __solve_SE(demands, temperatures, SE, cycles)
        except (MaximumIterationException, TemperatureProbagationException) as e:
            warnings.warn(f'Exception occured: {e}  \n ignore this sample and continue with next one')


# first run for d_sampling, doing all the tracing and stuff
def setup_d_sampling(d_dist, T_dist, SE, cycles):
    # set the SE-save-state to the state corresponding to the mean demand
    dem = d_dist.mean()
    temp = T_dist.mean()
    _solve_SE(dem, temp, SE, cycles)

def timing_d_sampling(n_samples, SE, cycles, d_dist, T_dist, verbose=False):
    demands = d_dist.sample(n_samples)
    temperatures = T_dist.sample(n_samples)

    print_frac = n_samples // 10
    for i, (dem, temp) in enumerate(zip(demands, temperatures)):
        if verbose and i % print_frac == 0:
            print(f'calculating sample {i}')
        # set demands
        SE.load_save_state()
        try:
            _solve_SE(dem, temp, SE, cycles)
        except MaximumIterationException:
            continue

def setup_mf_sampling(d_distribution, t_distribution, SE, cycles):
    ''' setup mass flow distribution '''
    d_mean = d_distribution.mean()
    d_cov = d_distribution.covariance()
    d_cor = np.transpose(d_cov / np.sqrt(np.diag(d_cov))) / np.sqrt(np.diag(d_cov))
    T_mean = t_distribution.mean()

    # calculate mass flow at mean demand and temperature
    _solve_SE(d_mean, T_mean, SE, cycles)
    mf_mean = np.zeros_like(d_mean)
    for d in SE.demands.keys():
        mf_mean[SE.dem_ind[d]] = SE.mf[SE.find_edge[d]].numpy()

    # calculate mf at demand mean + 1 sigma; mean temperature
    d_p1_std = d_mean + d_distribution.stddev()
    _solve_SE(d_p1_std, T_mean, SE, cycles)
    mf_p1_std = np.zeros_like(d_p1_std)
    for d in SE.demands.keys():
        mf_p1_std[SE.dem_ind[d]] = SE.mf[SE.find_edge[d]].numpy()

    mf_std = mf_p1_std - mf_mean
    mf_cov = tf.transpose(d_cor * mf_std) * mf_std

    # construct mf-distribution
    mf_dist = ZeroTruncatedMultivariateNormal(loc=tf.constant(mf_mean, dtype=tf.float64), scale_tril=tf.linalg.cholesky(mf_cov),
                                              validate_args=True, name='prior_mass_flow_distribution')
    return mf_dist

def timing_mf_sampling(n_samples, SE, cycles, grid, d_dist, T_dist, mf_dist, verbose=False):
    mf_samples = mf_dist.sample(n_samples)
    T_samples = T_dist.sample(n_samples)

    weights = np.zeros((n_samples, 2))

    print_frac = n_samples // 10

    # mfs = tf.Variable(np.zeros_like(SE.Q_heat))
    #     # FI.solve_p(SE, grid)
    for i, (mf, T) in enumerate(zip(mf_samples, T_samples)):
        while tf.reduce_sum(d_dist.signs * mf) < 0:
            mf = tf.squeeze(mf_dist.sample(1))
        if verbose and i % print_frac == 0:
            print(f'calculating sample {i}')
        SE.load_save_state()
        for dem in SE.demands.keys():
            ind = SE.dem_ind[dem]
            SE.set_active_edge_temperature(dem, T[ind])
        # set temperature for heating - last entry in T_vector
        ind = [k for k in SE.heatings.keys()][0]
        SE.set_active_edge_temperature(ind, T[-1])

        FI.solve_mf(SE, mf, cycles)
        FI.solve_p(SE, grid)
        FI.solve_temp(SE)

        # calculate demand from gridstate:
        Q_vals = SE.get_demand_from_grid(heating=False)
        weights[i, :] = [mf_dist.prob(mf_samples[i, :]), d_dist.prob(Q_vals)]

    w = weights[:, 1] / weights[:, 0]
    w = w / np.sum(w)

    eff_SR = 1 / (1 + np.var(w, ddof=1))
    n_eff = n_samples * eff_SR
    print(f'effective samplesize: {n_eff}')
    print(f'effective samplerate: {100 * eff_SR}%')
    return n_eff

class ImportanceSampler(object):

    def __init__(self, d_dist, T_dist, SE, cycles, grid):
        self.d_dist = d_dist
        self.T_dist = T_dist
        self.SE = SE
        self.cycles = cycles
        self.grid = grid

    def setup(self):
        self.mf_dist = setup_mf_sampling(self.d_dist, self.T_dist, self.SE, self.cycles)

    def generate_training_samples(self, n_samples, include_slack, file_spec, verbose=False):
        # alias for self.sample
        _ = self._sampling(n_samples, include_slack, verbose=verbose,
                           file_spec=file_spec, calc_weights=False, store_results=True)

    def _solve_single_sample(self, SE, mf, T):
        # only consider 'valid' samples in which the total consumption is larger than the total supply
        while tf.reduce_sum(self.d_dist.signs * mf) < 0:
            mf = tf.squeeze(self.mf_dist.sample(1))

        SE.load_save_state()
        for dem in SE.demands.keys():
            SE.set_active_edge_temperature(dem, T[SE.dem_ind[dem]])
        # set temperature for heating - last entry in T_vector
        ind = [k for k in SE.heatings.keys()][0]
        SE.set_active_edge_temperature(ind, T[-1])

        # solve sample
        FI.solve_mf(SE, mf, self.cycles)
        FI.solve_p(SE, self.grid)
        FI.solve_temp(SE)

    def _sampling(self, n_samples, include_slack, calc_weights=False, results_file=None, verbose=False):
        """
        performs IS sampling to generate training samples for the DNN
        :param n_samples:       number of samples to draw
        :param include_slack:   if True, the heating power is included in the demand output
        :param calc_weights:    if True, compute the sample weights
        :param results_file:    expects a function with one parameter returning the file to store the data in or None.
                                If None, no data is stored
        :param verbose:         if True, print progress to console
        :return:
        """

        # setup optional variables and alias:
        if calc_weights:
            weights = np.zeros((n_samples, 2))
        if verbose:
            print_frac = n_samples // 10
        store_results = results_file is not None

        # alias:
        SE = self.SE

        # run setup if not already done
        if not hasattr(self, 'mf_dist'):
            self.setup()

        ''' sampling process: '''
        # draw samples from distributions
        mf_samples = self.mf_dist.sample(n_samples)
        T_samples = self.T_dist.sample(n_samples)


        # solve each sample
        for i, (mf, T) in enumerate(zip(mf_samples, T_samples)):
            if verbose:
                if i % print_frac == 0:
                    print(f'calculating sample {i} of {n_samples} samples')

            self._solve_single_sample(SE, mf, T)

            if calc_weights:
                # calculate demand from gridstate:
                Q_vals = SE.get_demand_from_grid(heating=False)
                weights[i, :] = [self.mf_dist.prob(mf_samples[i, :]), self.d_dist.prob(Q_vals)]

            if store_results:
                # calculate demand from gridstate and save results:
                Q_vals = SE.get_demand_from_grid(heating=include_slack)
                state = tf.concat([SE.T, SE.mf, SE.p, SE.T_end], axis=0)
                tf.print(Q_vals, T_samples[i, :], tf.transpose(state), summarize=-1,
                         output_stream='file://' + f'{results_file(i)}')

        if calc_weights:
            w = weights[:, 1] / weights[:, 0]
            w = w / np.sum(w)

            n_eff = n_samples / (1 + np.var(w, ddof=1))
            print(f'effective samplesize: {n_eff}')
            print(f'effective samplerate: {100 / (1 + np.var(w, ddof=1))}%')

        return n_eff if calc_weights else np.nan



'''
def generate_samples(n_samples, SE, cycles, grid, d_dist, T_dist, res_file, heating=False, store_weights=False, verbose=False):
    """
    performs IS sampling to generate training samples for the DNN and write them into a result file

    :param n_samples: number of samples to generate
    :param SE: State-Equations object
    :param cycles: loops in the grid
    :param grid: grid model
    :param d_dist: demand distribution
    :param T_dist: temperature distribution
    :param res_file: file to write the results to
    :param heating: if True, the heating power is included in the demand output
    :param store_weights: if True, the not normalised weights of the samples are stored in the result file
    :param verbose: if True, the progress is printed to the console
    """

    mf_dist = setup_mf_sampling(d_dist, T_dist, SE, cycles)
    mf_samples = mf_dist.sample(n_samples)
    T_samples = T_dist.sample(n_samples)

    if store_weights or verbose:
        weights = np.zeros((n_samples, 2))

    for i, (mf, T) in enumerate(zip(mf_samples, T_samples)):
        # only consider 'valid' samples in which the total consumption is larger than the total supply
        while tf.reduce_sum(d_dist.signs * mf) < 0:
            mf = tf.squeeze(mf_dist.sample(1))
        if verbose:
            if i % 100 == 0:
                print(f'calculating sample {i}')

        # set active edge temperatures
        SE.load_save_state()
        for dem in SE.demands.keys():
            SE.set_active_edge_temperature(dem, T[SE.dem_ind[dem]])
        # set temperature for heating - last entry in T_vector
        ind = [k for k in SE.heatings.keys()][0]
        SE.set_active_edge_temperature(ind, T[-1])

        # solve sample
        FI.solve_mf(SE, mf, cycles)
        FI.solve_p(SE, grid)
        FI.solve_temp(SE)

        if store_weights or verbose:
            Q_vals = SE.get_demand_from_grid(heating=False)  # heating power acts as slack, not included in distr.
            weights[i, :] = [mf_dist.prob(mf_samples[i, :]), d_dist.prob(Q_vals)]

        # calculate demand from gridstate and save results:
        Q_vals = SE.get_demand_from_grid(heating=heating)
        state = tf.concat([SE.T, SE.mf, SE.p, SE.T_end], axis=0)
        if not store_weights:
            tf.print(Q_vals, T_samples[i, :], tf.transpose(state), summarize=-1,
                     output_stream='file://' + f'{res_file}.csv')
        else:
            tf.print(weights[i, 1]/weights[i,0], Q_vals, T_samples[i, :], tf.transpose(state), summarize=-1,
                     output_stream='file://' + f'{res_file}.csv')

    if verbose:
        w = weights[:, 1] / weights[:, 0]
        w = w / np.sum(w)

        n_eff = n_samples / (1 + np.var(w, ddof=1))
        print(f'effective samplesize: {n_eff}')
        print(f'effective samplerate: {100 / (1 + np.var(w, ddof=1))}%')
'''
