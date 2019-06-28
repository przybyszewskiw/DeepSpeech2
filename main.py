#!/usr/bin/env python3
import argparse
import runpy
import torch.cuda
from trainer import Trainer


def main():
    torch.set_printoptions(edgeitems=5)

    parser = argparse.ArgumentParser(description='DeepSpeech2 training')
    parser.add_argument('--config', type=str, default='./configs/default.py')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--local_rank', type=int, required=False)  # needed for launch of distributed training

    args = parser.parse_args()

    if args.cuda:
        if not torch.cuda.is_available():
            raise Exception("CUDA (GPU) is not available!")

    if args.local_rank is not None:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
    else:
        args.local_rank = 0

    config_module = runpy.run_path(args.config)

    run = Trainer(net_params=config_module.get('net_params'),
                  train_params=config_module.get('train_params'),
                  device='cuda' if args.cuda else 'cpu',
                  my_rank=args.local_rank)

    run.train()


if __name__ == '__main__':
    main()
