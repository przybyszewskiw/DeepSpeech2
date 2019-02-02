from model import DeepSpeech
import numpy as np
from dataload import load_track, load_transcript
from decoder import ctcBeamSearch
from jiwer import wer
import torch
import torch.optim as optim

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
    #print(transcript, answer)
    error = wer(transcript, answer)
    print("Current Word Error Rate (WER) is {}".format(error))
