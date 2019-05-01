import sys

import numpy as np
from decoder import ctcBeamSearch
from jiwer import wer
import torch
import ctcbeam


def eval_model(model, dataset, libri_dataset):
    error = 0
    if len(dataset) != len(libri_dataset):
        raise RuntimeError(
            "Datasets sizes not equal len(dataset):{}, len(libri_dataset):{}".format(len(dataset),
                                                                                     len(
                                                                                         libri_dataset)))
    with torch.no_grad():
        for (track_path, transcript), (track, _) in zip(dataset, libri_dataset):
            print('evaluating "{}" at {}'.format(transcript, track_path))

            _, probs = model(track)
            probs = probs.squeeze()
            list_aux = torch.split(probs, [1, 28], 1)
            probs = torch.cat((list_aux[1], list_aux[0]), 1)
            beamWidth = 200

            answer = ctcbeam.ctcbeam(probs.tolist(), "ngrams.txt", beamWidth)
            # answer = ctcBeamSearch(probs)

            word_error_rate = wer(transcript, answer)
            error += word_error_rate
            print('answer = "{}" WER = {}'.format(answer, word_error_rate))
            sys.stdout.flush()

    print("Word Error Rate after evaluation {}.".format(error / len(libri_dataset)))
