import sys
import numpy as np
import torch
import soundfile as sf
import scipy.signal as signal


class Loader:
    def __init__(self, bucket_size, time_length, time_overlap):
        self.bucket_size = bucket_size
        self.time_length = time_length
        self.time_overlap = time_overlap

    # arguments: file_path - path to music file (must be mono)
    #            bucket_size - size of frequency bucket in hz
    #            time_overlap - overlap of time intervals in ms
    def load_track(self, file_path, debug=False):
        data, sample_rate = sf.read(file_path)

        if debug: print("length of data: {}".format(len(data)))
        if debug: print("sample rate: {}".format(sample_rate))
        nperseg = int(round(sample_rate * self.time_length / 1e3))
        noverlap = int(round(sample_rate * self.time_overlap / 1e3))
        nfft = int(round(sample_rate / self.bucket_size))

        freqs, times, spec = signal.spectrogram(data, fs=sample_rate,
                                                window=signal.get_window('hann', nperseg),
                                                nperseg=nperseg,
                                                noverlap=noverlap,
                                                nfft=nfft)

        if debug: print("spectrogram done")
        if debug: print("number of frequency bins: {}".format(len(freqs)))
        if debug: print("size of frequency bin: {} Hz".format(freqs[1] - freqs[0]))
        if debug: print("number of time samples: {}".format(len(times)))
        if debug: print("step between time samples: {} ms".format((times[1] - times[0]) * 1e3))

        return spec

    def load_transcript(self, file_path):
        with open(file_path, 'r') as f:
            transcript = f.read()

        return transcript

    def convert_transcript(self, trans):
        def convert_char(c):
            if ord('a') <= ord(c) <= ord('z'):
                return ord(c) - ord('a') + 1
            elif c == ' ':
                return 27
            elif c == "'":
                return 28
            else:
                raise Exception("Transcript unknown character:" + str(c))
        return [convert_char(c) for c in trans.lower()]

    def load_tensors(self, trackpath, transcript):
        track = self.load_track(trackpath)
        transcript = self.convert_transcript(transcript)
        return torch.from_numpy(track[np.newaxis, :]).float(), torch.FloatTensor([transcript]).int()

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


def test():
    if len(sys.argv) < 2:
        print("give name of file")
    else:
        # spectrogram with 5Hz frequency bins ans 20ms time bins overlapping 10ms
        loader = Loader(5, 20, 10)
        print(loader.load_track(sys.argv[1], True))


if __name__ == '__main__':
    test()
