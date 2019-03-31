import sys

import numpy as np
from decoder import ctcBeamSearch
from jiwer import wer
import torch


def eval_model(model, dataset, loader):
    error = 0

    for track_path, transcript in dataset:
        print('evaluating "{}" at {}'.format(transcript, track_path))
        track = loader.load_track(track_path)
        track = torch.from_numpy(track[np.newaxis, :]).float()

        _, probs = model(track)
        probs = probs.squeeze()
        list_aux = torch.split(probs, [1, 28], 1)
        probs = torch.cat((list_aux[1], list_aux[0]), 1)

        answer = ctcBeamSearch(probs)
        word_error_rate = wer(transcript, answer)
        error += word_error_rate
        print('answer = "{}" WER = {}'.format(answer, word_error_rate))
        sys.stdout.flush()

    print("Word Error Rate after evaluation {}.".format(error / len(dataset)))


def eval_single(model, track_path, transcript_path, sound_bucket_size, sound_time_overlap):
    track = load_track(track_path, sound_bucket_size, sound_time_overlap)
    track = torch.from_numpy(track[np.newaxis, :]).float()
    with open(transcript_path, 'r') as ground_truth:
        transcript = ground_truth.read().replace('\n', '')

    _, probs = model(track)
    probs = probs.squeeze()
    list_aux = torch.split(probs, [1, 28], 1)
    probs = torch.cat((list_aux[1], list_aux[0]), 1)

    answer = ctcBeamSearch(probs)
    # print(transcript, answer)
    error = wer(transcript, answer)
    print("Current Word Error Rate (WER) is {}".format(error))
