from lrpolicy import LrPolicy as LrP
from torch.nn.init import xavier_normal_

base_params = {
    # ----- Convolutions -----
    'conv_layers': [
        {'kernel': (41, 11), 'stride': (2, 2), 'num_chan': 32},
        {'kernel': (21, 11), 'stride': (2, 1), 'num_chan': 64},
        {'kernel': (21, 11), 'stride': (2, 1), 'num_chan': 92},
    ],

    'frequencies': 160,

    # ----- Recurrent -----
    'rec_number': 3,

    # ----- FullyConnected -----
    'fc_layers_sizes': [2048],

    # ----- Others -----
    'lr_policy': LrP.EXP_DECAY,
    'lr_policy_params': {
        'lr': 0.0005,
        'decay_steps': 8788,  # number of iterations in epoch
        'decay_rate': 0.83,
        'min_lr': 0,
    },

    'batch_size': 32,

    'batch_norm': True,

    'dropout': 0.5,

    'epochs': 50,

    'shuffle_dataset': True,

    'model_saving_epoch': 1,

    # starting epoch will be sorted regardless of shuffle_dataset value
    'sorta_grad': True,

    'weights_initializer': 'xavier_normal',

    'l2_regularization_scale': 0
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

    'workers': 20,
}
