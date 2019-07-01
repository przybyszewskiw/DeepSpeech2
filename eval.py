#!/usr/bin/env python3
import torch
import argparse
from evaluator import eval_on_dataset, eval_on_tracks
from model import DeepSpeech
from scripts.librispeech import LibriSpeech


def main():
    torch.set_printoptions(edgeitems=5)

    parser = argparse.ArgumentParser(description='DeepSpeech2 evaluation')
    parser.add_argument('--track-dir', type=str, required=False)
    parser.add_argument('--dataset', type=str, required=False)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--beam-width', default=200)
    parser.add_argument('--alpha', default=0.1)
    parser.add_argument('--beta', default=0.0)
    parser.add_argument('--lm-file', type=str, default='./src/4-gram.binary')
    parser.add_argument('--trie-file', type=str, default='./src/librispeech-probs.txt')

    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    net = DeepSpeech(**(checkpoint['net_params']))
    net.load_state_dict(checkpoint['state_dict'])

    if args.cuda:
        net = net.cuda()

    if args.dataset is not None:
        eval_on_dataset(net, LibriSpeech().get_dataset(args.dataset), checkpoint['net_params'], args)
    if args.track_dir is not None:
        eval_on_tracks(net, args.track_dir, checkpoint['net_params'], args)


if __name__ == '__main__':
    main()
