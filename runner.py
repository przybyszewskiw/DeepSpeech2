import os
import numpy as np
import torch
import torch.optim as optim
from dataload import load_track, load_transcript, convert_transcript
from evaluator import eval_single, eval_model
from model import DeepSpeech
from scripts.librispeech import LibriSpeech


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

    def train_single(self, track_path, transcript_path):
        transcript = load_transcript(transcript_path)
        track, transcript = self.get_tensors(track_path, transcript)

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
    """
    def train_epoch(self, dataset):
        for (i, (track_path, transcript_string)) in enumerate(dataset):
            track, transcript = self.load_tensors(track_path, transcript_string)
            self.optimizer.zero_grad()
            output, probs = self.net(track)
            loss = DeepSpeech.criterion(output, transcript)
            print("loss in {}th iteration is {}".format(i, loss))
    """

    def train_epoch(self, dataset, batch_size=8):
        tracks_to_merge = []
        for (i, (track_path, transcript_string)) in enumerate(dataset):
            if (i + 1) % batch_size != 0:
                tracks_to_merge.append(self.load_tensors(track_path, transcript_string))
            else:
                (audio, transs, lengths) = self.merge_into_batch(tracks_to_merge)
                tracks_to_merge = []
                self.optimizer.zero_grad()
                output, _ = self.net(audio)

                loss = DeepSpeech.criterion(output, transs, lengths)
                loss.backward()
                self.optimizer.step()
                print("loss in {}th iteration is {}".format(i, loss.item()))

    def train(self, dataset, epochs=70):
        for epoch in range(epochs):
            if not os.path.isdir("./models"):
                print("Creating a directory for saved models")
                os.makedirs("./models")
            torch.save(self.net.state_dict(), "./models/{}-iters.pt".format(epoch))
            print(epoch)
            if epoch % 5 == 4:
                torch.save(self.net.state_dict(), "./models/{}-iters.pt".format(epoch))
                self.net.eval()
                eval_model(self.net, dataset, self.sound_bucket_size, self.sound_time_overlap)
            self.net.train()
            self.train_epoch(dataset)

    def merge_into_batch(self, tracks):
        dim1 = tracks[0][0].shape[1]
        dim2 = max([tensor.shape[2] for tensor, _ in tracks])
        extended_audio_tensors = [
            torch.cat(
                [tensor, torch.zeros(1, dim1, dim2 - tensor.shape[2])],
                dim=2
            ) for tensor, _ in tracks
        ]

        lengths_tensor = torch.FloatTensor([trans.shape[1] for _, trans in tracks]).int()
        transs_tensor = torch.cat([trans for _, trans in tracks], dim=1).squeeze()
        audio_tensor = torch.cat(extended_audio_tensors, dim=0)

        return audio_tensor, transs_tensor, lengths_tensor

    def load_tensors(self, trackpath, transcript):
        track = load_track(trackpath, self.sound_bucket_size, self.sound_time_overlap)
        transcript = convert_transcript(transcript)
        return torch.from_numpy(track[np.newaxis, :]).float(), torch.FloatTensor([transcript]).int()


def test2():
    r = Runner()
    # r.train_single('test_track.wav', 'test_transcript.txt')
    r.train(LibriSpeech().get_dataset('test-clean'))


def test():
    r = Runner()
    tracks = LibriSpeech().get_dataset('test-clean')[:16]
    track_tens = [r.load_tensors(ph, ts) for ph, ts in tracks]
    (audio, transs, lengths) = r.merge_into_batch(track_tens)
    print(tracks)
    print(audio.shape)
    print(transs)
    print(lengths)


if __name__ == "__main__":
    torch.set_printoptions(edgeitems=5)
    test()
    test2()
