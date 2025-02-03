"""
This is a config file to specify the structure of the neural network used in the experiments and the training procedure.
"""

#%% Neural network specification:
DNN_layout = {
    'layers': [200, 400, 400],      # number of neurons in the trainable inner layers
    'include_slack_output': True,   # include power value of the slack power plant in the output vector
    'include_slack_input': False    # include power value of the slack power plant in the input vector
    }
#%% training settings:
n_train_MSE = 50_000                # Number of training samples for the MSE training
n_val = 10_000                      # Number of validation samples used for early stopping in the MSE training
n_train_PAL = 10_000                # Number of training samples for the PAL training
n_test = 10_000                     # Number of test samples for model evaluation

# training settings for MSE training
train_settings_MSE = {
    'loss': 'MSE',
    'power MSE': False,
    'optimizer': 'Adam',
    'opt_config': {},
    'batch_size': 32,
    'epochs': 5_000,
    'n_train': n_train_MSE,
    'n_val': n_val,
    'save model': False,
    'early stopping': 50,
    'start_epoch': 0,
}

# training settings for PAL training
train_settings_PAL = {
    'loss': 'PAL',
    'optimizer': 'Adadelta',
    'opt_config': {'learning_rate': 0.01},
    'batch_size': 32,
    'epochs': 2000,
    'n_train': n_train_PAL,
    'early stopping': 20,
    'save model': False
}