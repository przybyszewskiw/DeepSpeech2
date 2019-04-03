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
    parser.add_argument('--track', type=str, required=False)
    parser.add_argument('--model', type=str, required=False)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--starting-epoch', type=int, default=0)
    parser.add_argument('--device', type=str, required=False, default='cpu', choices=['gpu', 'cpu'])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--batch-norm', dest='batch_norm', action='store_true')
    parser.set_defaults(batch_norm=False)

    args = parser.parse_args()

    if args.device == 'gpu':
        if not torch.cuda.is_available():
            raise Exception("CUDA (GPU) is not available!")

    if args.model is not None:
        run = Runner(pretrained_model_path=args.model,
                     device=args.device,
                     batch_norm=args.batch_norm)
    else:
        run = Runner(device=args.device,
                     batch_norm=args.batch_norm)

    if args.task == 'train':
        if args.dataset is None:
            raise Exception("Specify dataset to train on!")
        run.train(dataset=LibriSpeech().get_dataset(args.dataset),
                  epochs=args.epochs,
                  starting_epoch=args.starting_epoch,
                  batch_size=args.batch_size)

    elif args.task == 'eval':
        if args.model is None:
            raise Exception('Specify model to evaluate!')

        if args.dataset is not None:
            run.eval_on_dataset(LibriSpeech().get_dataset(args.dataset))
        elif args.track is not None:
            assert 'Not implemented yet!'
        else:
            raise Exception('Nothing to evaluate on!')


if __name__ == '__main__':
    main()

