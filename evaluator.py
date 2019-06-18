import sys

import numpy as np
from decoder import ctcBeamSearch
from jiwer import wer
import torch
import ctcbeam
import random

def eval_model(model, dataset, libri_dataset, beam_width, lm_file):
    error = 0
    if len(dataset) != len(libri_dataset):
        raise RuntimeError(
            "Datasets sizes not equal len(dataset):{}, len(libri_dataset):{}".format(len(dataset),
                                                                                     len(libri_dataset)))
    sum_length = 0
    sum_error = 0
    report_freq = 50
    sum_wer = 0
    beam_width = 200
    random.seed(12345)
    random.shuffle(dataset)
    model.eval()
    print("Length of dataset: {}".format(len(dataset)))
    with torch.no_grad():
        index = 0
        for (track_path, transcript), (track, _) in zip(dataset, libri_dataset):
            index += 1
            print('evaluating "{}" at {}'.format(transcript, track_path))

            _, probs = model(track)
            probs = probs.squeeze()
            list_aux = torch.split(probs, [1, 28], 1)
            probs = torch.cat((list_aux[1], list_aux[0]), 1)

            answer = ctcbeam.ctcbeam(probs.tolist(),
                                     lm_file,
                                     "../librispeech-vocab-probs.txt",
                                     beam_width,
                                     0.1,
                                     0.0)

            word_error_rate = wer(transcript, answer)
            sum_wer += word_error_rate
            length = len(transcript.split())
            sum_error += word_error_rate * length
            sum_length += length
            print('answer = "{}" WER = {}'.format(answer, word_error_rate))
            if (index % report_freq == 0):
                print("Running WER after {} examples: {}, {} / {}; average: {}".format(index, sum_error / sum_length, sum_error, sum_length, sum_wer / index))
            sys.stdout.flush()

    print("Word Error Rate after evaluation: {}.".format(sum_error / sum_length))


def eval_tracks(model, tracks, dataset, lm_file):
    with torch.no_grad():
        model.eval()
        for track, (track_ten, _) in zip(tracks, dataset):
            print('evaluating {}'.format(track))

            _, probs = model(track_ten)
            probs = probs.squeeze()
            list_aux = torch.split(probs, [1, 28], 1)
            probs = torch.cat((list_aux[1], list_aux[0]), 1)
            beamWidth = 200

            answer = ctcbeam.ctcbeam(probs.tolist(),
                                     lm_file,
                                     "../librispeech-vocab-probs.txt",
                                     beamWidth,
                                     0.05,
                                     0.05)

            print('evaluated answer = "{}"'.format(answer))
            sys.stdout.flush()
