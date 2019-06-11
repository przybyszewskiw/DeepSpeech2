import numpy as np
import torch
import soundfile as sf
import scipy.signal as signal
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


def ctc_collate_fn(tracks):
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


def normalize_signal(signal):
    return signal / (np.max(np.abs(signal)) + 1e-5)


def get_libri_dataloader(dataset, batch_size=1, shuffle=False, num_workers=0):
    return DataLoader(dataset, batch_size, shuffle, num_workers=num_workers,
                      collate_fn=ctc_collate_fn)


class AudioDataset(Dataset):
    def __init__(self, ls_dataset, num_audio_features, time_length, time_overlap, eps=1e-10):
        super(Dataset, self)
        self.time_length = time_length
        self.time_overlap = time_overlap
        self.num_audio_features = num_audio_features
        self.eps = eps
        self.ls_dataset = ls_dataset

    def __len__(self):
        return len(self.ls_dataset)

    def __getitem__(self, idx):
        return self._load_tensors(*self.ls_dataset[idx])

    # arguments: file_path - path to music file (must be mono)
    #            bucket_size - size of frequency bucket in hz
    #            time_overlap - overlap of time intervals in ms
    def _load_track(self, file_path, debug=False, eps=0.0001):
        data, sample_rate = sf.read(file_path)
        data = normalize_signal(data)

        if debug: print("length of data: {}".format(len(data)))
        if debug: print("sample rate: {}".format(sample_rate))
        nperseg = int(round(sample_rate * self.time_length / 1e3))
        noverlap = int(round(sample_rate * self.time_overlap / 1e3))

        freqs, times, spec = signal.spectrogram(data, fs=sample_rate,
                                                window=signal.get_window('hann', nperseg),
                                                nperseg=nperseg,
                                                noverlap=noverlap,
                                                detrend=False)

        spec = np.square(np.abs(spec))
        spec[spec <= 1e-30] = 1e-30
        spec = 10 * np.log10(spec)

        assert self.num_audio_features <= spec.shape[0]
        spec = spec[:self.num_audio_features, :]

        mean = np.mean(spec, axis=0)
        std_dev = np.std(spec, axis=0)
        spec = (spec - mean) / np.sqrt(std_dev ** 2 + eps)

        if debug: print("spectrogram done")
        if debug: print("number of frequency bins: {}".format(spec.shape[0]))
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

    def _load_tensors(self, track_path, transcript):
        transcript = self.convert_transcript(transcript)
        trans_ten = torch.FloatTensor([transcript]).int()

        track = self._load_track(track_path)
        track_ten = torch.from_numpy(track[np.newaxis, :]).float()

        return track_ten, trans_ten


