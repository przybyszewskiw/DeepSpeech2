from model import DeepSpeech
import numpy as np
from dataload import load_track, load_transcript, convert_char
from evaluator import eval_single, eval_model
from scripts.librispeech import LibriSpeech
import torch
import torch.optim as optim


class Runner:
    def __init__(self, frequencies=1601,
                 conv_number=2,
                 context=5,
                 rec_number=3,
                 full_number=2,
                 characters=29,
                 sound_bucket_size=5,
                 sound_time_overlap=5,
                 lr=0.01):
        self.net = DeepSpeech(frequencies=frequencies,
                              conv_number=conv_number,
                              context=context,
                              rec_number=rec_number,
                              full_number=full_number,
                              characters=characters)
        self.sound_bucket_size = sound_bucket_size
        self.sound_time_overlap = sound_time_overlap
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr)

    # TODO can be extended to work with many batches
    @staticmethod
    def get_tensors(track, transcript):
        return torch.from_numpy(track[np.newaxis, :]).float(), torch.FloatTensor([transcript]).int()

    def train_single(self, track_path, transcript_path):
        track = load_track(track_path, self.sound_bucket_size, self.sound_time_overlap)
        transcript = load_transcript(transcript_path)

        track, transcript = self.get_tensors(track, transcript)
        for epoch in range(100):
            self.optimizer.zero_grad()
            output, probs = self.net(track)
            loss = DeepSpeech.criterion(output, transcript)
            print("loss={}".format(loss))
            if epoch % 20 == 19:
                eval_single(self.net, track_path, transcript_path,
                            self.sound_bucket_size, self.sound_time_overlap)
            loss.backward()
            self.optimizer.step()

    """
        dataset - list of pairs (track_path, transcrip_string)
    """
    def train_epoch(self, dataset):
        for (i, (track_path, transcrip_string)) in enumerate(dataset):
            track = load_track(track_path, self.sound_bucket_size, self.sound_time_overlap)
            transcript = [convert_char(c) for c in transcrip_string]

            track, transcript = self.get_tensors(track, transcript)
            self.optimizer.zero_grad()
            output, probs = self.net(track)
            loss = DeepSpeech.criterion(output, transcript)
            print("loss in {}th iteration is {}".format(i, loss))

    def train(self, dataset, epochs=2):
        for epoch in range(epochs):
            print(epoch)
            if epoch % 2 == 1:
                self.net.eval()
                eval_model(self.net, dataset, self.sound_bucket_size, self.sound_time_overlap)
            self.net.train()
            self.train_epoch(dataset)

def test():
    r = Runner()
    #r.train_single('test_track.wav', 'test_transcript.txt')
    r.train(LibriSpeech().get_dataset('test-clean'))


if __name__ == "__main__":
    torch.set_printoptions(edgeitems = 20)
    test()
