from lrpolicy import LrPolicy as LrP
from torch.nn.init import xavier_normal_
from scripts.librispeech import LibriSpeech


net_params = {
    # ----- Convolutions -----
    'conv_layers': [
        {'kernel': (41, 11), 'stride': (2, 2), 'num_chan': 32},
        {'kernel': (21, 11), 'stride': (2, 1), 'num_chan': 64},
        {'kernel': (21, 11), 'stride': (2, 1), 'num_chan': 92},
    ],

    # ----- Recurrent -----
    'rec_number': 3,
    'rec_type': 'lstm',
    'rec_bidirectional': True,

    # ----- FullyConnected -----
    'fc_layers_sizes': [2048],

    'batch_norm': True,

    'fc_dropout': 0.5,

    # ----- Spectogram ------
    'sound_features_size': 160,
    'sound_time_overlap': 5,
    'sound_time_length': 20,

    'characters': 29,

    'frequencies': 160,
}


train_params = {
    'lr_policy': LrP.POLY_DECAY,
    'lr_policy_params': {
        'lr': 0.0002,
        'decay_steps': 1000,
        'power': 0.5,
        'min_lr': 0,
    },

    'l2_regularization_scale': 0,

    'weights_initializer': xavier_normal_,

    'epochs': 50,

    'batch_size': 32,

    # choices: None, 'O0', 'O1', 'O2', 'O3'
    'amp_opt_level': 'O1',

    'shuffle_dataset': True,

    # how often shall we save checkpoint
    'model_saving_epoch': 1,

    # starting epoch will be sorted regardless of shuffle_dataset value
    'sorta_grad': True,

    'workers': 20,

    'train_dataset': LibriSpeech().get_dataset('test-clean'),

    'test_dataset': LibriSpeech().get_dataset('test-clean'),

    'models_dir': './models'
}