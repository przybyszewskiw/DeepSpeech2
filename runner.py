import datetime
import glob
import math
import os
import sys
import time
import torch
import torch.optim as optim
import torch.nn as nn
import dataload as dl
from evaluator import eval_model, eval_tracks
from model import DeepSpeech
import lrpolicy as lrp
import runpy
import json  # TODO delete -- only for dict printing

try:
    import apex
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    print('APEX not found, install it if you want to train in mixed precision')


class Runner:
    def __init__(self, config_path,
                 pretrained_model_path=None,
                 device='cpu',
                 my_rank=0
                 ):
        self.my_rank = my_rank
        config_module = runpy.run_path(config_path)
        self.base_params = config_module.get('base_params')
        self.adv_params = config_module.get('adv_params')
        self.non_json_params = config_module.get('non_json_params')

        print("Loaded config file from {}".format(config_path))
        print("base_params:", json.dumps(self.base_params, indent=4))
        print("adv_params:", json.dumps(self.adv_params, indent=4))

        self.net = DeepSpeech(conv_initial_channels=self.base_params["frequencies"],
                              conv_layers=self.base_params["conv_layers"],
                              rec_number=self.base_params["rec_number"],
                              rec_type=self.base_params["rec_type"],
                              rec_bidirectional=self.base_params["rec_bidirectional"],
                              fc_layers_sizes=self.base_params["fc_layers_sizes"],
                              characters=self.adv_params["characters"],
                              batch_norm=self.base_params["batch_norm"],
                              fc_dropout=self.base_params["dropout"],
                              initializer=self.non_json_params[
                                  self.base_params["weights_initializer"]],
                              flatten=(self.base_params['mixed_precision_opt_level'] is not None))

        if device == 'gpu':
            device = 'cuda'

        if 'WORLD_SIZE' not in os.environ and self.base_params['mixed_precision_opt_level'] is not None:
            raise Exception('Use distributed parallelism to train in mixed precision!')

        if 'WORLD_SIZE' in os.environ and self.base_params['mixed_precision_opt_level'] is None:
            raise Exception('Dont use distributed parallelism to train in normal precision!')

        if self.base_params['mixed_precision_opt_level'] is not None:
            print("using apex synced BN")
            self.net = apex.parallel.convert_syncbn_model(self.net)

        self.device = device
        self.net = self.net.to(self.device)

        self.lr = self.base_params["lr_policy_params"]["lr"]
        self.l2_regularization_scale = self.base_params["l2_regularization_scale"]
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=self.lr,
                                    weight_decay=self.l2_regularization_scale)
        self.optimizer_steps = 0

        if self.base_params['mixed_precision_opt_level'] is not None:
            self.net, self.optimizer = amp.initialize(
                self.net, self.optimizer,
                opt_level=self.base_params['mixed_precision_opt_level'])

        self.is_data_paralel = False

        if device == 'cuda':
            if self.base_params['mixed_precision_opt_level'] is None:
                self.net = nn.DataParallel(self.net)
                self.is_data_paralel = True
            else:
                self.net = DDP(self.net, delay_allreduce=True)

        devicce = torch.device('cpu')
        if pretrained_model_path is not None:
            self.net.load_state_dict(torch.load(pretrained_model_path, map_location=devicce))

    """
        dataset - list of pairs (track_path, transcript_string)
    """

    def train_epoch(self, dataloader):
        self.net.train()
        total_loss = 0.
        skipped = 0

        for i, (audio, transs, lengths) in enumerate(dataloader):
            start_time = time.time()
            self.optimizer.zero_grad()

            audio = audio.to(self.device)
            output, _ = self.net(audio)

            if self.device != torch.device("cpu"):
                output = output.to("cpu")

            loss = DeepSpeech.criterion(output, transs, lengths)

            if loss == float('inf'):
                print(
                    "WARNING: loss is inf in {}th iteration, omitting track".format(
                        i), file=sys.stderr)
                skipped += 1
                continue

            if math.isnan(loss.item()):
                print(
                    "WARNING: loss is nan in {}th iteration, omitting track".format(
                        i))
                skipped += 1
                continue

            print("[{}] loss in {}th iteration is {}, it took {} seconds".format(
                datetime.datetime.now(),
                i,
                loss.item(),
                time.time() - start_time
            ))

            total_loss += loss.item()

            if self.base_params['mixed_precision_opt_level'] is not None:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()
            self.optimizer_steps += 1

            # for batch_size around 32 in total
            lr_policy = self.base_params["lr_policy"]
            lr_policy_params = self.base_params["lr_policy_params"]

            lrp.apply_policy(self.optimizer, self.optimizer_steps, lr_policy, lr_policy_params)

            # for some reason output is in the buffer until termination while redirecting to file,
            # so we have to manually flush
            sys.stdout.flush()

        print('[{}] Total loss in this epoch is {}'.format(datetime.datetime.now(), total_loss / (len(dataloader) - skipped)))

    def test_dataset(self, dataloader):
        self.net.eval()
        total_loss = 0.
        iterations = 0
        for i, (audio, transs, lengths) in enumerate(dataloader):
            audio = audio.to(self.device)
            output, _ = self.net(audio)

            if self.device != torch.device("cpu"):
                output = output.to("cpu")

            loss = DeepSpeech.criterion(output, transs, lengths)

            total_loss += loss.item()
            iterations += 1

        print('[{}] Validation loss is {}'.format(datetime.datetime.now(), total_loss / iterations))

    def train(self, dataset, testing_dataset=None, model_save_pth='./models'):
        starting_epoch = self.adv_params["starting_epoch"]
        epochs = self.base_params["epochs"]
        batch_size = self.base_params["batch_size"]
        model_saving_epoch = self.base_params["model_saving_epoch"]
        shuffle_dataset = self.base_params["shuffle_dataset"]
        sorta_grad = self.base_params['sorta_grad']
        workers = self.adv_params["workers"]

        libri_dataset = self._get_libri_dataset(dataset)

        libri_testing_dataset = self._get_libri_dataset(testing_dataset)

        self.net.train()
        for epoch in range(starting_epoch, epochs):
            if not os.path.isdir(model_save_pth):
                print("Creating a directory for saved modeldevices")
                os.makedirs(model_save_pth)
            print(epoch)
            start_time = time.time()

            test_sampler = None

            if self.base_params['mixed_precision_opt_level'] is not None:
                train_sampler = torch.utils.data.distributed.DistributedSampler(libri_dataset)
                test_sampler = torch.utils.data.distributed.DistributedSampler(libri_testing_dataset)

                libri_dataloader = dl.get_libri_dataloader(
                    libri_dataset,
                    batch_size=batch_size,
                    num_workers=workers,
                    sampler=train_sampler
                )

            else:
                if shuffle_dataset and not (sorta_grad and epoch == starting_epoch):
                    libri_dataloader = dl.get_libri_dataloader(
                        libri_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=workers
                    )
                else:
                    libri_dataloader = dl.get_libri_dataloader(
                        libri_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=workers
                    )

            self.train_epoch(libri_dataloader)
            print('[{}] Training {}. epoch took {} seconds'.format(datetime.datetime.now(), epoch, time.time() - start_time))

            if testing_dataset is not None:
                if self.base_params['mixed_precision_opt_level'] is None:
                    with torch.no_grad():
                        libri_testing_dataloader = dl.get_libri_dataloader(libri_testing_dataset,
                                                                           batch_size=batch_size)
                        self.test_dataset(libri_testing_dataloader)
                else:
                    libri_testing_dataloader = dl.get_libri_dataloader(
                        libri_testing_dataset,
                        batch_size=batch_size,
                        sampler=test_sampler
                    )
                    self.test_dataset(libri_testing_dataloader)

            if epoch % model_saving_epoch == model_saving_epoch - 1 and self.my_rank == 0:
                print('Saving model')
                file_path = os.path.join(model_save_pth, "{}-epoch.pt".format(epoch))
                if self.is_data_paralel:
                    torch.save(self.net.module.state_dict(), file_path)
                else:
                    torch.save(self.net.state_dict(), file_path)

    def eval_on_dataset(self, dataset, lm_file):
        self.net.eval()
        libri_dataset = self._get_libri_dataset(dataset)
        beam_width = self.adv_params["beam_width"]

        eval_model(self.net, dataset, libri_dataset, beam_width, lm_file)

    def eval_on_tracks(self, dir, lm_file):
        self.net.eval()
        tracks = glob.glob(os.path.join(dir, '*.flac'))
        print('Evaluating {}'.format(tracks))
        dataset = self._get_tracks_dataset(tracks)
        eval_tracks(self.net, tracks, dataset, lm_file)

    def _get_tracks_dataset(self, tracks):
        dataset = [(tr, "") for tr in tracks]
        return dl.AudioDataset(dataset, num_audio_features=self.adv_params["sound_features_size"],
                               time_overlap=self.adv_params["sound_time_overlap"],
                               time_length=self.adv_params["sound_time_length"])

    def eval_on_tracks(self, dir):
        self.net.eval()
        tracks = glob.glob(os.path.join(dir, '*.flac'))
        print('Evaluating {}'.format(tracks))
        dataset = self._get_tracks_dataset(tracks)
        eval_tracks(self.net, tracks, dataset)

    def _get_tracks_dataset(self, tracks):
        dataset = [(tr, "") for tr in tracks]
        return dl.AudioDataset(dataset, num_audio_features=self.adv_params["sound_features_size"],
                               time_overlap=self.adv_params["sound_time_overlap"],
                               time_length=self.adv_params["sound_time_length"])

    def _get_libri_dataset(self, dataset):
        return dl.AudioDataset(dataset, num_audio_features=self.adv_params["sound_features_size"],
                               time_overlap=self.adv_params["sound_time_overlap"],
                               time_length=self.adv_params["sound_time_length"])
