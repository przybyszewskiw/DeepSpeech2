import os
from os.path import join as pjoin, abspath
import shutil
import subprocess
import tempfile
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

    def _decompress_dataset(self, fname):
        subprocess.check_call(['tar', 'zxvf', fname, '-C', '../datasets/'])

    def _parse_librispeech_root(self, root):
        res = []
        for root, _, files in os.walk(abspath(root)):
            for file in files:
                if file.split('.')[-1] == 'txt':
                    res += self._parse_librispeech_file(pjoin(root, file))
        return res

    def _download_dataset(self, dataset_url):
        tmpdir = tempfile.mkdtemp()
        path = pjoin(tmpdir, 'dataset.tar.gz')
        urllib.request.urlretrieve(dataset_url, pjoin(tmpdir, path))
        self._decompress_dataset(path)
        shutil.rmtree(tmpdir)

    def get_dataset(self, name):
        dataset_root = abspath(pjoin('../datasets/LibriSpeech', name))
        if not os.path.isdir(dataset_root):
            self._download_dataset(DATASETS[name])
        return self._parse_librispeech_root(dataset_root)

    def get_all_datasets(self):
        res = []
        for k in DATASETS.keys():
            res += self.get_dataset(k)
        return res

    def get_clean_datasets(self):
        res = []
        for k in DATASETS.keys():
            if 'clean' in k:
                res += self.get_dataset(k)
        return res


if __name__ == '__main__':
    print(LibriSpeech().get_dataset('test-clean'))
