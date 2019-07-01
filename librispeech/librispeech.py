import os
from os.path import join as pjoin, abspath
import shutil
import subprocess
import tempfile
import soundfile as sf
import urllib.request

DATASETS = {
    'test-clean': 'http://www.openslr.org/resources/12/test-clean.tar.gz',
    'test-other': 'http://www.openslr.org/resources/12/test-other.tar.gz',
    'train-clean-100': 'http://www.openslr.org/resources/12/train-clean-100.tar.gz',
    'train-clean-360': 'http://www.openslr.org/resources/12/train-clean-360.tar.gz',
    'train-other-500': 'http://www.openslr.org/resources/12/train-other-500.tar.gz',
}


class LibriSpeech:
    def _parse_librispeech_file(self, trans_file):
        res = []
        dir = os.path.dirname(trans_file)
        with open(trans_file, 'r') as f:
            for line in f:
                splitted = line.split()
                filename = '{}.flac'.format(splitted[0])
                transcription = ' '.join(splitted[1:])
                res.append((pjoin(dir, filename), transcription))
        return res

    def _get_lenght(self, trackpath):
        track = sf.SoundFile(trackpath)
        return len(track) / track.samplerate

    def _sort_dataset(self, dataset):
        with_lenght = [(self._get_lenght(path), path, trans) for path, trans in dataset]
        with_lenght = sorted(with_lenght)
        return [(path, trans) for _, path, trans in with_lenght]

    def _parse_librispeech_root(self, root):
        res = []
        for root, _, files in os.walk(abspath(root)):
            for file in files:
                if file.split('.')[-1] == 'txt':
                    res += self._parse_librispeech_file(pjoin(root, file))
        return res

    def _download_dataset(self, name, dataset_url):
        print('Dataset not found! Downloading {}'.format(name))
        shutil.rmtree(pjoin('./datasets', name + 'tar.gz'))
        subprocess.check_call(['wget', '-P', './datasets/', dataset_url])
        if not os.path.isdir("./datasets"):
            print("Creating a directory for datasets")
            os.makedirs("./datasets")
        subprocess.check_call(['tar', 'zxvf', pjoin('./datasets', name + '.tar.gz'), '-C', './datasets/'])

    def _get_dataset(self, name):
        if name == 'all-train':
            return self.get_all_train_datasets()
        if name == 'all-train-clean':
            return self.get_all_train_clean_datasets()

        dataset_root = abspath(pjoin('./datasets/LibriSpeech', name))
        if not os.path.isdir(dataset_root):
            self._download_dataset(name, DATASETS[name])
        dataset = self._parse_librispeech_root(dataset_root)
        return dataset

    def get_dataset(self, name, sort=True):
        return self._get_datasets([name], sort)

    def _get_datasets(self, names, sort=True):
        res = []
        for name in names:
            res += self._get_dataset(name)
        return self._sort_dataset(res) if sort else res

    def get_all_datasets(self):
        return self._get_datasets(DATASETS.keys())

    def get_all_train_datasets(self):
        return self._get_datasets(['train-clean-100', 'train-clean-360', 'train-other-500'])

    def get_all_train_clean_datasets(self):
        return self._get_datasets(['train-clean-100', 'train-clean-360'])

    def get_clean_datasets(self):
        return self._get_datasets([name for name in DATASETS.keys() if 'clean' in name])

