import sys
import soundfile as sf
import scipy.signal as signal


# arguments: file_path - path to music file (must be mono)
#            bucket_size - size of frequency bucket in hz
#            time_overlap - overlap of time intervals in ms
def load_track(file_path, bucket_size, time_overlap):
    data, sample_rate = sf.read(file_path)

    print("length of data: {}".format(len(data)))
    print("sample rate: {}".format(sample_rate))
    nperseg = int(round(sample_rate / bucket_size))
    noverlap = int(round(sample_rate / bucket_size - time_overlap * sample_rate / 1e3))
    
    freqs, times, spec = signal.spectrogram(data, fs=sample_rate,
                                            window='hann',
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            detrend=False)

    print("spectrogram done")
    print("number of frequency bins: {}".format(len(freqs)))
    print("size of frequency bin: {} Hz".format(freqs[1] - freqs[0]))
    print("number of time samples: {}".format(len(times)))
    print("size of time sample: {} ms".format(times[1] - times[0]))
    
    return spec


def load_transcript(file_path):
    def convert_char(c):
        if ord('a') <= ord(c) <= ord('z'):
            return ord(c) - ord('a') + 1
        elif c == ' ':
            return 27
        elif c == '\'':
            return 28
        else:
            raise Exception("Transcript unknown character:" + str(c))

    with open(file_path, 'r') as f:
        transcript = f.read()
    transcript_num = [convert_char(c) for c in transcript[:-1]]

    return transcript_num


if len(sys.argv) < 2:
    print("give name of file")
else:
    # spektrogram co 5hz z czasami długości 10ms, z overlapami 5ms
    print(load_track(sys.argv[1], 5, 5).shape)
