from lrpolicy import LrPolicy as LrP
from torch.nn.init import xavier_normal_

base_params = {
    # ----- Convolutions -----
    'conv_layers': [{'kernel': (11, 41), 'stride': (1, 1), 'num_chan': 1}],

    'frequencies': 160,

    # ----- Recurrent -----
    'rec_number': 2,

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

    'weights_initializer': 'xavier_normal'
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

    'starting_epoch': 0
}
