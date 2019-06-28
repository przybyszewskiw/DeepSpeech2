from lrpolicy import LrPolicy as LrP
from torch.nn.init import xavier_normal_

base_params = {
    # ----- Convolutions -----
    'conv_layers': [{'kernel': (11, 41), 'stride': (1, 1), 'num_chan': 1}],

    'frequencies': 160,

    # ----- Recurrent -----
    'rec_number': 2,
    'rec_type': 'rnn',
    'rec_bidirectional': True,

    # ----- FullyConnected -----
    'fc_layers_sizes': [2048],

    # ----- Others -----
    'lr_policy': LrP.NO_DECAY,
    'lr_policy_params': {
        'lr': 0.0001,
    },

    'batch_size': 32,

    'batch_norm': False,

    'dropout': 0,

    'epochs': 1,

    'shuffle_dataset': True,

    'model_saving_epoch': 5,

    # starting epoch will be sorted regardless of shuffle_dataset value
    'sorta_grad': False,

    'weights_initializer': 'xavier_normal',

    'l2_regularization_scale': 0,

    # possibilities: 'O0', 'O1', 'O2', 'O3'
    # more detailed description of each level
    # of optimization can be found here:
    # https://nvidia.github.io/apex/amp.html
    'mixed_precision_opt_level': 'O1'
}

non_json_params = {
    'xavier_normal': xavier_normal_
}

adv_params = {
    # ----- Spectogram ------
    'sound_features_size': 160,
    'sound_time_overlap': 5,
    'sound_time_length': 20,

    'characters': 29,

    'starting_epoch': 0,

    'workers': 0,

    'beam_width': 200,
    'alpha': 0.1,
    'beta': 0.0
}