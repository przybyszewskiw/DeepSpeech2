import os
import random
import sys
import time
import torch
import torch.optim as optim
import torch.nn as nn
from dataload import Loader
from evaluator import eval_single, eval_model
from model import DeepSpeech
from scripts.librispeech import LibriSpeech
import lrpolicy as lrp
import runpy
import json  # TODO delete -- only for dict printing

shandom_ruffle = random.shuffle


class Runner:
    def __init__(self, config_path,
                 pretrained_model_path=None,
                 device='cpu'
                 ):
        config_module = runpy.run_path(config_path)
        self.base_params = config_module.get('base_params')
        self.adv_params = config_module.get('adv_params')

        print("Loaded config file from {}".format(config_path))
        print("base_params:", json.dumps(self.base_params, indent=4))
        print("adv_params:", json.dumps(self.adv_params, indent=4))

        self.net = DeepSpeech(conv_initial_channels=self.base_params["frequencies"],
                              conv_layers=self.base_params["conv_layers"],
                              rec_number=self.base_params["rec_number"],
                              fc_layers_sizes=self.base_params["fc_layers_sizes"],
                              characters=self.adv_params["characters"],
                              batch_norm=self.base_params["batch_norm"],
                              fc_dropout=self.base_params["dropout"])

        if device == 'gpu':
            device = 'cuda:0'
            self.net = nn.DataParallel(self.net)

        self.device = torch.device(device)
        self.net = self.net.to(self.device)

        if pretrained_model_path is not None:
            self.net.load_state_dict(torch.load(pretrained_model_path))

        self.loader = Loader(num_audio_features=self.adv_params["sound_features_size"],
                             time_overlap=self.adv_params["sound_time_overlap"],
                             time_length=self.adv_params["sound_time_length"])

        self.lr = self.base_params["lr_policy_params"]["lr"]
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=self.lr,
                                    betas=(0.9, 0.999))
        self.optimizer_steps = 0

    def train_single(self, track_path, transcript_path):
        transcript = self.loader.load_transcript(transcript_path)
        track, transcript = self.loader.load_tensors(track_path, transcript)

        for epoch in range(100):
            self.optimizer.zero_grad()
            output, probs = self.net(track)
            loss = DeepSpeech.criterion(output, transcript)
            print("loss={}".format(loss))
            if epoch % 20 == 19:
                eval_single(self.net, track_path, transcript_path,
                            self.loader)
            loss.backward()
            self.optimizer.step()

    """
        dataset - list of pairs (track_path, transcript_string)
    """

    def train_epoch(self, dataset, batch_size=8):
        self.net.train()
        tracks_to_merge = []
        total_loss = 0.
        iterations = 0
        for (i, (track_path, transcript_string)) in enumerate(dataset):
            tracks_to_merge.append(self.loader.load_tensors(
                track_path,
                transcript_string
            ))

            if (i + 1) % batch_size == 0:
                start_time = time.time()
                (audio, transs, lengths) = self.loader.merge_into_batch(tracks_to_merge)
                tracks_to_merge = []
                self.optimizer.zero_grad()

                audio = audio.to(self.device)
                output, _ = self.net(audio)

                if self.device != torch.device("cpu"):
                    print("moving net output to CPU")
                    output = output.to("cpu")

                print("Starting criterion calculation")
                loss = DeepSpeech.criterion(output, transs, lengths)
                loss.backward()
                self.optimizer.step()
                self.optimizer_steps += 1

                # for batch_size around 32 in total
                lr_policy = self.base_params["lr_policy"]
                lr_policy_params = self.base_params["lr_policy_params"]

                lrp.apply_policy(self.optimizer, self.optimizer_steps, lr_policy, lr_policy_params)

                print("loss in {}th iteration is {}, it took {} seconds".format(
                    i,
                    loss.item(),
                    time.time() - start_time
                ))
                total_loss += loss.item()
                iterations += 1
                # for some reason output is in the buffer until termination while redirecting to file,
                # so we have to manually flush
                sys.stdout.flush()

        print('Total loss in this epoch is {}'.format(total_loss / iterations))

    def test_dataset(self, dataset, batch_size=8):
        self.net.eval()
        tracks_to_merge = []
        total_loss = 0.
        iterations = 0
        with torch.no_grad():
            for (i, (track_path, transcript_string)) in enumerate(dataset):
                tracks_to_merge.append(self.loader.load_tensors(
                    track_path,
                    transcript_string
                ))

                if (i + 1) % batch_size == 0:
                    (audio, transs, lengths) = self.loader.merge_into_batch(tracks_to_merge)
                    tracks_to_merge = []

                    audio = audio.to(self.device)
                    output, _ = self.net(audio)

                    if self.device != torch.device("cpu"):
                        output = output.to("cpu")

                    loss = DeepSpeech.criterion(output, transs, lengths)

                    total_loss += loss.item()
                    iterations += 1

        print('Validation loss is {}'.format(total_loss / iterations))

    def train(self, dataset, testing_dataset=None):
        starting_epoch = self.adv_params["starting_epoch"]
        epochs = self.base_params["epochs"]
        batch_size = self.base_params["batch_size"]
        model_saving_epoch = self.base_params["model_saving_epoch"]
        shuffle_dataset = self.base_params["shuffle_dataset"]

        self.net.train()
        for epoch in range(starting_epoch, epochs):
            if not os.path.isdir("./models"):
                print("Creating a directory for saved modeldevices")
                os.makedirs("./models")
            print(epoch)
            start_time = time.time()
            if shuffle_dataset:
                shandom_ruffle(dataset)
            self.train_epoch(dataset, batch_size=batch_size)
            print('Training {}. epoch took {} seconds'.format(epoch, time.time() - start_time))

            if testing_dataset is not None:
                self.test_dataset(testing_dataset)

            if epoch % model_saving_epoch == model_saving_epoch - 1:
                print('Saving model')
                torch.save(self.net.state_dict(), "./models/{}-iters.pt".format(epoch))

    def eval_on_dataset(self, dataset):
        self.net.eval()
        eval_model(self.net, dataset, self.loader)
