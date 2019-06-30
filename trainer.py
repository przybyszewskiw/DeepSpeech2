import datetime
import math
import os
import sys
import time
import torch
import torch.optim as optim
import torch.nn as nn
import dataload as dl
from model import DeepSpeech
import lrpolicy as lrp

try:
    import apex
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    print('APEX not found, install it if you want to train in mixed precision')


class Trainer:
    def __init__(self,
                 train_params,
                 net_params,
                 device='cpu',
                 my_rank=0,
                 checkpoint=None
                 ):
        self.my_rank = my_rank
        self.train_params = train_params
        self.net_params = net_params

        self.net = DeepSpeech(flatten=train_params['amp_opt_level'] is not None,
                              initializer=train_params['weights_initializer'],
                              **net_params)

        if checkpoint is not None:
            checkpoint_file = torch.load(checkpoint, map_location='cpu')
            self.net.load_state_dict(checkpoint_file['state_dict'])

        if 'WORLD_SIZE' not in os.environ and train_params['amp_opt_level'] is not None:
            raise Exception('Use distributed parallelism to train in mixed precision!')

        if 'WORLD_SIZE' in os.environ and train_params['amp_opt_level'] is None:
            raise Exception('Dont use distributed parallelism to train in normal precision!')

        if train_params['amp_opt_level'] is not None:
            print("using apex synced BN")
            self.net = apex.parallel.convert_syncbn_model(self.net)

        self.device = device
        self.net = self.net.to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=train_params['lr_policy_params']['lr'],
                                    weight_decay=train_params['l2_regularization_scale'])
        self.optimizer_steps = 0
        self.starting_epoch = 0

        if checkpoint is not None:
            self.starting_epoch = checkpoint_file['epoch']
            self.optimizer_steps = checkpoint_file['optimizer_steps']
            self.optimizer.load_state_dict(checkpoint_file['optimizer_sd'])

        if train_params['amp_opt_level'] is not None:
            self.net, self.optimizer = amp.initialize(
                self.net, self.optimizer,
                opt_level=train_params['amp_opt_level'])

        self.is_data_paralel = False

        if device == 'cuda':
            if train_params['amp_opt_level'] is None:
                self.net = nn.DataParallel(self.net)
                self.is_data_paralel = True
            else:
                self.net = DDP(self.net, delay_allreduce=True)
                self.is_data_paralel = True

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

            if self.train_params['amp_opt_level'] is not None:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()

            if 'WORLD_SIZE' in os.environ:
                self.optimizer_steps += int(os.environ['WORLD_SIZE'])
            else:
                self.optimizer_steps += 1

            lr_policy = self.train_params["lr_policy"]
            lr_policy_params = self.train_params["lr_policy_params"]

            lrp.apply_policy(self.optimizer, self.optimizer_steps, lr_policy, lr_policy_params)

            sys.stdout.flush()

        print('[{}] Total loss in this epoch is {}'.format(datetime.datetime.now(),
                                                           total_loss / (len(dataloader) - skipped)))

    def test_epoch(self, dataloader):
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

    def train(self):
        epochs = self.train_params["epochs"]
        batch_size = self.train_params["batch_size"]
        model_saving_epoch = self.train_params["model_saving_epoch"]
        shuffle_dataset = self.train_params["shuffle_dataset"]
        sorta_grad = self.train_params['sorta_grad']
        workers = self.train_params["workers"]
        models_dir = self.train_params["models_dir"]
        train_dataset = dl.AudioDataset(self.train_params['train_dataset'], self.net_params)
        test_dataset = dl.AudioDataset(self.train_params['test_dataset'], self.net_params)
        self.train_params["lr_policy_params"]['max_iter'] = len(train_dataset) * epochs / batch_size

        self.net.train()
        for epoch in range(self.starting_epoch, epochs):
            if not os.path.isdir(models_dir):
                print("Creating a directory for saved models")
                os.makedirs(models_dir)
            print(epoch)
            start_time = time.time()

            if self.train_params['amp_opt_level'] is not None:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
                test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

                train_dl = dl.get_dataloader(
                    train_dataset,
                    batch_size=batch_size,
                    num_workers=workers,
                    sampler=train_sampler
                )

            else:
                if shuffle_dataset and not sorta_grad:
                    train_dl = dl.get_dataloader(
                        train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=workers
                    )
                else:
                    train_dl = dl.get_dataloader(
                        train_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=workers
                    )

            self.train_epoch(train_dl)
            print('[{}] Training {}. epoch took {} seconds'.format(datetime.datetime.now(), epoch + 1,
                                                                   time.time() - start_time))

            if self.train_params['amp_opt_level'] is None:
                with torch.no_grad():
                    test_dl = dl.get_dataloader(test_dataset, batch_size=batch_size)
                    self.test_epoch(test_dl)
            else:
                test_dl = dl.get_dataloader(
                    test_dataset,
                    batch_size=batch_size,
                    sampler=test_sampler
                )
                self.test_epoch(test_dl)

            if (epoch + 1) % model_saving_epoch == 0 and self.my_rank == 0:
                print('Saving model after epoch {}'.format(epoch + 1))
                if self.is_data_paralel:
                    state_dict = self.net.module.state_dict()
                else:
                    state_dict = self.net.state_dict()

                torch.save({
                    'epoch': epoch + 1,
                    'net_params': self.net_params,
                    'state_dict': state_dict,
                    'optimizer_steps': self.optimizer_steps,
                    'optimizer_sd': self.optimizer.state_dict()
                }, os.path.join(models_dir, "{}-epoch.pt".format(epoch + 1)))
