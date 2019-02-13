import sys
import soundfile as sf
import scipy.signal as signal


# arguments: file_path - path to music file (must be mono)
#            bucket_size - size of frequency bucket in hz
#            time_overlap - overlap of time intervals in ms
def load_track(file_path, bucket_size, time_overlap, debug=False):
    data, sample_rate = sf.read(file_path)

    if debug: print("length of data: {}".format(len(data)))
    if debug: print("sample rate: {}".format(sample_rate))
    nperseg = int(round(sample_rate / bucket_size))
    noverlap = int(round(sample_rate / bucket_size - time_overlap * sample_rate / 1e3))

    freqs, times, spec = signal.spectrogram(data, fs=sample_rate,
                                            window='hann',
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            detrend=False)

    if debug: print("spectrogram done")
    if debug: print("number of frequency bins: {}".format(len(freqs)))
    if debug: print("size of frequency bin: {} Hz".format(freqs[1] - freqs[0]))
    if debug: print("number of time samples: {}".format(len(times)))
    if debug: print("size of time sample: {} ms".format(times[1] - times[0]))

    return spec


def load_transcript(file_path):
    with open(file_path, 'r') as f:
        transcript = f.read()

    return transcript


def convert_transcript(trans):
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


def test():
    if len(sys.argv) < 2:
        print("give name of file")
    else:
        # spektrogram co 5hz z czasami długości 10ms, z overlapami 5ms
        print(load_track(sys.argv[1], 5, 5).shape)


if __name__ == '__main__':
    test()
