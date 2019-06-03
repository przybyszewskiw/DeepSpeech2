from lrpolicy import LrPolicy as LrP
from torch.nn.init import xavier_normal_

base_params = {
    # ----- Convolutions -----
    'conv_layers': [
        {'kernel': (11, 41), 'stride': (1, 1), 'num_chan': 32},
        {'kernel': (11, 21), 'stride': (1, 1), 'num_chan': 32},
        {'kernel': (11, 21), 'stride': (1, 1), 'num_chan': 32}
    ],

    'frequencies': 160,

    # ----- Recurrent -----
    'rec_number': 3,
    'rec_type': 'rnn',
    'rec_bidirectional': True,

    # ----- FullyConnected -----
    'fc_layers_sizes': [2048],

    # ----- Others -----
    'lr_policy': LrP.POLY_DECAY,
    'lr_policy_params': {
        'lr': 0.0002,
        'decay_steps': 1000,
        'power': 0.5,
        'min_lr': 0,
        'max_iter': int(281215 * 50 / 32)  # TODO add correct numbers of iterations
        # right now {~number of iterations on all-train} * {epochs} / {batch_size}
    },

    'batch_size': 32,

    'batch_norm': True,

    'dropout': 0.5,

    'epochs': 50,

    'shuffle_dataset': True,

    'model_saving_epoch': 2,

    # starting epoch will be sorted regardless of shuffle_dataset value
    'sorta_grad': False,

    'weights_initializer': 'xavier_normal',

    'l2_regularization_scale': 0,

    'mixed_precision_opt_level': None

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
}