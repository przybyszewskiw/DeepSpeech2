import numpy as np
import torch
import soundfile as sf
import scipy.signal as signal
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


def ctc_collate_fn(tracks):
    dim1 = tracks[0][0].shape[1]
    dim2 = max([tensor.shape[2] for tensor, _ in tracks])
    dim3 = min([tensor.shape[2] for tensor, _ in tracks])
    ratio = dim3 / dim2
    if ratio < 0.05:
        print("WARNING: Ratio of lengths in minibatch is {}".format(ratio))
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


def get_dataloader(dataset, batch_size=1, shuffle=False, num_workers=0, **kwargs):
    return DataLoader(dataset, batch_size, shuffle, num_workers=num_workers,
                      collate_fn=ctc_collate_fn, **kwargs)


# Dataset takes list of pairs (transcription, path) representing tracks
class AudioDataset(Dataset):
    def __init__(self, ls_dataset, num_audio_features, time_length, time_overlap, eps=0.0001):
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

    def _normalize_signal(self, signal):
        return signal / (np.max(np.abs(signal)) + 1e-5)

    # Works only with mono files
    def _load_track(self, file_path):
        data, sample_rate = sf.read(file_path)
        data = self._normalize_signal(data)

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
        spec = (spec - mean) / np.sqrt(std_dev ** 2 + self.eps)

        return spec

    def _load_transcript(self, file_path):
        with open(file_path, 'r') as f:
            transcript = f.read()

        return transcript

    def _convert_transcript(self, trans):
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
        transcript = self._convert_transcript(transcript)
        trans_ten = torch.FloatTensor([transcript]).int()

        track = self._load_track(track_path)
        track_ten = torch.from_numpy(track[np.newaxis, :]).float()

        return track_ten, trans_ten


