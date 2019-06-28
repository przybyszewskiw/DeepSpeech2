import glob
import os
import sys
from jiwer import wer
import torch
import ctcbeam
import random
from dataload import AudioDataset


def eval_on_dataset(model, track_list, net_params, args):
    sum_length = 0
    sum_error = 0
    report_freq = 50
    sum_wer = 0
    random.seed(12345)
    random.shuffle(track_list)
    dataset = AudioDataset(track_list, net_params)

    with torch.no_grad():
        index = 0
        for (track_path, transcript), (track, _) in zip(track_list, dataset):
            index += 1
            print('evaluating "{}" at {}'.format(transcript, track_path))

            answer = eval_track(model, track, args)
            word_error_rate = wer(transcript, answer)
            sum_wer += word_error_rate
            length = len(transcript.split())
            sum_error += word_error_rate * length
            sum_length += length

            print('answer = "{}" WER = {}'.format(answer, word_error_rate))

            if index % report_freq == 0:
                print("Running WER after {} examples: {}, {} / {}; average: {}".format(
                    index, sum_error / sum_length, sum_error, sum_length, sum_wer / index))
            sys.stdout.flush()

    print("Word Error Rate after evaluation: {}.".format(sum_error / sum_length))


def eval_track(model, tensor, args):
    with torch.no_grad():
        model.eval()
        _, probs = model(tensor)
        probs = probs.squeeze()
        list_aux = torch.split(probs, [1, 28], 1)
        probs = torch.cat((list_aux[1], list_aux[0]), 1)

        answer = ctcbeam.ctcbeam(probs.tolist(),
                                 args.lm_file,
                                 args.trie_file,
                                 args.beam_width,
                                 args.alpha,
                                 args.beta)
        return answer


def eval_on_tracks(net, dir, net_params, args):
    tracks = glob.glob(os.path.join(dir, '*.flac'))
    dataset = AudioDataset([(tr, "") for tr in tracks], net_params)

    with torch.no_grad():
        net.eval()
        for track, (track_ten, _) in zip(tracks, dataset):
            print('evaluating {}'.format(track))
            answer = eval_track(net, track_ten, args)
            print('evaluated answer = "{}"'.format(answer))
            sys.stdout.flush()
