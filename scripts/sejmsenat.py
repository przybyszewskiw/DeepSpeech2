import os
import random


def parse_file(path):
    res = []
    with open(path, 'r') as f:
        for line in f:
            words = line.split()
            trans = ' '.join(words[1:])
            fname = words[0]
            dirname = '-'.join(fname.split('-')[:-1])
            track_name = fname.split('-')[-1] + '.wav'
            track_path = os.path.join(dirname, track_name)
            res.append((track_path, trans))
    random.shuffle(res)
    return res


class SejmSenat:
    VALID = './datasets/SejmSenat/test/text'
    TRAIN = './datasets/SejmSenat/train/text'

    def append_path(self, path):
        return os.path.join('./datasets', 'SejmSenat', 'audio', path)

    def get_train(self):
        return [(self.append_path(p), trans) for (p, trans) in parse_file(self.TRAIN)]

    def get_valid(self):
        return [(self.append_path(p), trans) for (p, trans) in parse_file(self.VALID)]


# if __name__ == '__main__':
#     dt = SejmSenat()
#     print(dt.get_valid())
