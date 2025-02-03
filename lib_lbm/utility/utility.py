"""
This file defines different utilities for classes in the SESAUU_tasks library.

Tasks:

get_scenario_params(scenario_name, output_file='auto'): wrapper for load_scenario function defined in _load_scenario
"""

import os
import re
from ._load_scenario import load_scenario

#%% Section 1: Tasks
"""
Tasks: The experiments are split up into different "tasks", i.e. generating training data, training the network, 
       which all follow the structure defined in the Task class. 
DNNTask / DNNData: defines additional structures to handle training data and model weights  
"""

class Task(object):
    def __init__(self, scenario_params=None, name=None, verbose=False, **kwargs):
        self.name = name if name is not None else self.__class__.__name__
        self.verbose = verbose if not hasattr(self, 'verbose') else self.verbose
        # set attributes for Values defined in scenario_params
        if scenario_params is not None:
            self._set_scenario_attributes(scenario_params)

    def __str__(self):
        return self.name

    def _set_scenario_attributes(self, scenario_params):
        for key, value in scenario_params.items():
            setattr(self, key, value)
        scenario_name = scenario_params['scenario_name']
        self._training_data_path = f'data_files/{scenario_name}/'
        self.training_data_file = lambda idx: f'{self._training_data_path}{scenario_name}_{idx}.csv'

    def execute(self):
        # executes the task - return a dict of parameters to be passed to later tasks
        return dict()

    def report(self, *args):
        if len(args) == 1:
            self._report(*args)
        else:
            self._report('')

    def _report(self, message):
        # reports the results to the screen or a result file
        if self.output_file is None:
            print(message)
        else:
            with open(self.output_file, 'a') as f:
                f.write(message)


class DNNTask(Task):
    ''' subclass for all tasks which work with the neural network '''
    def __init__(self, DataHandler=None, DNN=None, **kwargs):
        super().__init__(**kwargs)
        self.DNN = DNN
        self.Data = DataHandler

    def model_weight_file(self, spec=None):
        if spec is None:
            return f'./model_logs/{self.scenario_name}/{self.scenario_name}'
        elif type(spec) is int:
            return f'./model_logs/{self.scenario_name}/{self.scenario_name}_{spec:04}'
        else:
            return f'./model_logs/{self.scenario_name}/{self.scenario_name}_{spec}'


#%% Section 2: Scenario handling
"""
This section defines different classes and functions to create, and load scenarios
get_scenario_params: public interface to _load_scenario function defined in _load_scenario
create_scenario: creates a new grid file
"""

def get_scenario_params(scenario_name, output_file='auto'):
    grid, cycles, SE, d_dist, T_dist, = load_scenario(scenario_name)
    if output_file == 'auto':
        output_dir = f'results/{scenario_name}'
        output_file = f'{output_dir}/numerical_results.out'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    elif output_file is not None:
        output_dir = '/'.join(output_file.split('/')[:-1])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = None

    return {'grid': grid,
            'cycles': cycles,
            'SE': SE,
            'd_dist': d_dist,
            'T_dist': T_dist,
            'output_dir': output_dir,
            'output_file': output_file,
            'scenario_name': scenario_name}


def create_scenario(testcase, pp_pos=[0], grid_dim=10.):
    kind, n_actives = re.match(r"([a-z]+)([0-9]+)", testcase, re.I).groups()
    n_actives = int(n_actives)
    if len(pp_pos) < 1:
        raise Exception('At least one pp_pos must be specified but none were given')

    from .generate_grid import generate_lader_grid, generate_cycle_grid, print_grid_to_file, auto_set_iso_id

    if kind == 'ladder':
        [agn, agp, ae, aecp, aecn] = generate_lader_grid(n_rungs=n_actives, pp_pos=pp_pos, x_grid_distance=grid_dim,
                                                         y_grid_distance=10, iso_id='Isoplus_Std_DN40')
    elif kind == 'cycle':
        [agn, agp, ae, aecp, aecn] = generate_cycle_grid(n_sections=n_actives, pp_pos=pp_pos, rad_grid_distance=grid_dim,
                                                         iso_id='Isoplus_Std_DN40')
    else:
        raise Exception('unknown specification for grid kind')

    # print grid to file as this is the easiest way to convert it into a MeFlex-Grid-object
    print_grid_to_file(f'grids/{testcase}.json', agn, agp, ae, aecp, aecn)
    auto_set_iso_id(testcase, agp, aecp)
    print_grid_to_file(f'grids/{testcase}.json', agn, agp, ae, aecp, aecn)
