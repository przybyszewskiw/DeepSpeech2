#!/usr/bin/env python3
import argparse
import torch.cuda
from runner import Runner
from scripts.librispeech import LibriSpeech


def main():
    torch.set_printoptions(edgeitems=5)

    parser = argparse.ArgumentParser(description='DeepSpeech2!')
    parser.add_argument('task', action='store', choices=['train', 'eval'])
    parser.add_argument('--dataset', type=str, required=False)
    parser.add_argument('--config', type=str, default='./configs/default.py')
    parser.add_argument('--test-dataset', type=str, default='test-clean')
    parser.add_argument('--track-dir', type=str, required=False)
    parser.add_argument('--model', type=str, required=False)
    parser.add_argument('--device', type=str, required=False, default='cpu', choices=['gpu', 'cpu'])

    args = parser.parse_args()

    if args.device == 'gpu':
        if not torch.cuda.is_available():
            raise Exception("CUDA (GPU) is not available!")

    if args.model is not None:
        run = Runner(config_path=args.config,
                     pretrained_model_path=args.model,
                     device=args.device)
    else:
        run = Runner(
            config_path=args.config,
            device=args.device)

    if args.task == 'train':
        if args.dataset is None:
            raise Exception("Specify dataset to train on!")
        ls = LibriSpeech()
        run.train(dataset=ls.get_dataset(args.dataset),
                  testing_dataset=ls.get_dataset(args.test_dataset))

    elif args.task == 'eval':
        if args.model is None:
            raise Exception('Specify model to evaluate!')

        if args.dataset is not None:
            run.eval_on_dataset(LibriSpeech().get_dataset(args.dataset))
        elif args.track_dir is not None:
            run.eval_on_tracks(args.track_dir)
        else:
            raise Exception('Nothing to evaluate on!')


if __name__ == '__main__':
    main()
