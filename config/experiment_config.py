"""
This is a config file to specify the experiments run in Experiments.ipynb

To add a new test case, add a new entry to the test_cases list.
    'scenario': either 'ladder' or 'cycle' followed by the number of active edges
    'pp_pos': list of pp positions
    'grid_dim': grid dimension -> distance between two active edges in meters

To add a new experiment, define a corresponding function in the experiments section.
    The function should return a list of tuples (tasks, {task parameter})  to be executed in SESAAU_results_pipeline.py.
    The tasks are defined in lib_lbm.tasks.py and are subclasses of Task.
    Each experiment should start with the SetupScenario task which creates all necessary files for the scenario.
    Running the ResetOutput task will delete all output files for the scenario.

@Andreas Bott 2023
"""

#%% imports
import lib_lbm.tasks as tasks
import config.DNN_config as DNN_config


#%% Experiments:
def get_experiment_IS_sampling(testcase, repeat=5, n_samples=None):
    if n_samples is None:
        n_samples = [10, 100, 1000, 5000, 10_000]
    # setup header:
    experiment = [
        (tasks.SetupScenario, {**testcase, **{'create_new': 'auto'}}),
        (tasks.ResetOutputFile, {'output_file': f'results/{testcase["scenario"]}/IS_sampling.out'}),
    ]
    # add tasks for each entry in n_samples, repeat each task repeat times:
    for n in n_samples:
        for _ in range(repeat):
            experiment.append((tasks.CompareSampleTimes, {'n_samples': n}))
    return experiment

def get_experiment_Generate_training_samples(testcase):
    experiment = [
        (tasks.SetupScenario, {**testcase, **{'create_new': 'auto',
                                              'output_file': f'results/{testcase["scenario"]}/training_samples.out'}}),
        (tasks.ResetOutputFile, {}),
        (tasks.GenerateTrainingSamples, {'n_samples': DNN_config.n_train_MSE + DNN_config.n_val,
                                         'include_slack': DNN_config.DNN_layout['include_slack_input'] or
                                                          DNN_config.DNN_layout['include_slack_output'],
                                         'ignore_existing': True,
                                         'name': 'Training and Validation Samples'})
    ]
    # include_slack: whether to add a power value for the slack power plant to the training data
    # ignore_existing: whether to overwrite existing training data
    #   True: the task will generate the specified number of training samples
    #   False: the task will generate training samples until the specified number of training samples is reached
    return experiment

def get_experiment_Generate_test_samples(testcase):
    experiment = [
        (tasks.SetupScenario, {**testcase, **{'create_new': 'auto',
                                              'output_file': f'results/{testcase["scenario"]}/training_samples.out'}}),
        (tasks.ResetOutputFile, {}),
        (tasks.GenerateTrainingSamples, {'n_samples': DNN_config.n_test,
                                         'include_slack': DNN_config.DNN_layout['include_slack_input'] or
                                                          DNN_config.DNN_layout['include_slack_output'],
                                         'ignore_existing': True,
                                         'name': 'Training and Validation Samples'})
    ]
    # include_slack: whether to add a power value for the slack power plant to the training data
    # ignore_existing: whether to overwrite existing training data
    #   True: the task will generate the specified number of training samples
    #   False: the task will generate training samples until the specified number of training samples is reached
    return experiment

def get_experiment_Train_models_WMSE(testcase):
    experiment = [
        (tasks.SetupScenario, {**testcase, **{'create_new': 'auto',
                                              'output_file': f'results/{testcase["scenario"]}/training_samples.out'}}),
        (tasks.ResetOutputFile, {}),
        (tasks.BuildDNN, DNN_config),
        (tasks.TrainDNN, {'start_epoch': 0,
                          'verbose': 'auto',
                          'training_settings': DNN_config.train_settings_MSE,
                          'n_test_samples': DNN_config.n_test,
                          'name': 'WMSE Training'})
    ]
    # start_epoch: epoch to start training from (0: start from scratch)
    # verbose: verbosity level (0: no output, 1: progress bar, 2: full output) or 'auto' to use the default verbosity
    # training_settings: dictionary of training settings (see above)
    # n_test_samples: number of test samples to use for model evaluation
    return experiment

def get_experiment_Train_models_PAL(testcase):
    experiment = [
        (tasks.SetupScenario, {**testcase, **{'create_new': 'auto',
                                              'output_file': f'results/{testcase["scenario"]}/training_samples.out'}}),
        (tasks.ResetOutputFile, {}),
        (tasks.BuildDNN, DNN_config),
        (tasks.TrainDNN, {'start_epoch': 0,
                          'verbose': 'auto',
                          'training_settings': DNN_config.train_settings_PAL,
                          'n_test_samples': DNN_config.n_test,
                          'name': 'PAL Training'})
    ]
    # start_epoch: epoch to start training from (0: start from scratch)
    # verbose: verbosity level (0: no output, 1: progress bar, 2: full output) or 'auto' to use the default verbosity
    # training_settings: dictionary of training settings (see above)
    # n_test_samples: number of test samples to use for model evaluation
    return experiment
