from model import DeepSpeech
import numpy as np
from dataload import load_track, load_transcript
from decoder import ctcBeamSearch
import torch
import torch.optim as optim

class Runner:
    def __init__(self, frequencies=1103,
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
            if epoch % 5 == 0:
                probs = probs.squeeze()
                list_aux = torch.split(probs, [1, 28], 1)
                probs = torch.cat((list_aux[1], list_aux[0]), 1)
                print(probs)
                print(ctcBeamSearch(probs))
            loss.backward()
            self.optimizer.step()


def test():
    r = Runner()
    r.train_single('test_track.wav', 'test_transcript.txt')


if __name__ == "__main__":
    torch.set_printoptions(edgeitems = 20)
    test()
