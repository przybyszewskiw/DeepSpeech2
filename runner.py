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

shandom_ruffle = random.shuffle

class Runner:
    def __init__(self, frequencies=100,
                 rec_number=2,
                 full_layers=[1024],
                 characters=29,
                 sound_features_size=100,
                 sound_time_overlap=5,
                 sound_time_length=20,
                 lr=0.0001,
                 pretrained_model_path=None,
                 device='cpu',
                 batch_norm=False,
                 dropout=0):
        self.net = DeepSpeech(frequencies=frequencies,
                              rec_number=rec_number,
                              full_layers=full_layers,
                              characters=characters,
                              batch_norm=batch_norm,
                              fc_dropout=dropout)
        if device == 'gpu':
            device = 'cuda:0'
            self.net = nn.DataParallel(self.net)

        self.device = torch.device(device)
        self.net = self.net.to(self.device)

        if pretrained_model_path is not None:
            self.net.load_state_dict(torch.load(pretrained_model_path))

        self.loader = Loader(num_audio_features=sound_features_size,
                             time_overlap=sound_time_overlap,
                             time_length=sound_time_length)

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, betas=(0.9, 0.999))
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
                if self.optimizer_steps % 5000 == 4999:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.9
                        if param_group['lr'] < 0:
                            param_group['lr'] = 0

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

    def train(self, dataset, batch_size=8, epochs=50, starting_epoch=0, shuffle=False, testing_dataset=None):
        self.net.train()
        for epoch in range(starting_epoch, epochs):
            if not os.path.isdir("./models"):
                print("Creating a directory for saved modeldevices")
                os.makedirs("./models")
            print(epoch)
            start_time = time.time()
            if shuffle:
                shandom_ruffle(dataset)
            self.train_epoch(dataset, batch_size=batch_size)
            print('Training {}. epoch took {} seconds'.format(epoch, time.time() - start_time))

            if testing_dataset is not None:
                self.test_dataset(testing_dataset)

            if epoch % 5 == 4:
                print('Saving model')
                torch.save(self.net.state_dict(), "./models/{}-iters.pt".format(epoch))
                # Takes too much time!
                # self.net.eval()
                # eval_model(self.net, dataset, self.sound_bucket_size, self.sound_time_overlap)
                # self.net.train()

    def eval_on_dataset(self, dataset):
        self.net.eval()
        eval_model(self.net, dataset, self.loader)


def test_training():
    r = Runner(pretrained_model_path='models/4-iters.pt')
    r.train(
        dataset=LibriSpeech().get_dataset('test-clean'),
        epochs=100,
        starting_epoch=5
    )


def test_eval():
    r = Runner(pretrained_model_path='models/4-iters.pt')
    r.eval_on_dataset(LibriSpeech().get_dataset('test-clean', sort=False))


def test():
    r = Runner()
    tracks = LibriSpeech().get_dataset('test-clean')[:16]
    track_tens = [r.loader.load_tensors(ph, ts) for ph, ts in tracks]
    (audio, transs, lengths) = r.loader.merge_into_batch(track_tens)
    print(tracks)
    print(audio.shape)
    print(transs)
    print(lengths)


if __name__ == "__main__":
    torch.set_printoptions(edgeitems=5)
