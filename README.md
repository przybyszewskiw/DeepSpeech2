# Deep Speech 2

Implementation of the deep neural network for automatic speech recognition, based on [”Deep Speech 2: End-to-End Speech Recognition in English and Mandarin”](https://arxiv.org/pdf/1512.02595.pdf).

## Prerequisites
* Python >=v3.5
* [Pytorch](https://pytorch.org/) >=v1.0
* [Apex AMP](https://nvidia.github.io/apex/amp.html) - if you wish to train in mixed precision

## Getting Started
Installing python dependencies.
```
pip install -r requirements.txt
```
Preparing library responsible for beam search.
```
cd src/ && ./setup_ctcbeam_and_lm.sh
```
This script will download Kenlm and Librispeech language models and then compile C++ submodule into python-callable library.
## Training
#### Training configuration
Please see `configs/` subdirectory for configuration files. Feel free to modify them in your own experiments.
#### Training in fixed 32-bit precision
Please set in configuration file the value of `amp_opt_level` to `None`.
```
./main.py --cuda --config PATH/TO/CONFIG
```
To resume training from the checkpoint please use `--resume-training` flag to specify location of that checkpoint.
Note that you can resume training with different `train_params`, but `net_params` should remain the same.
Both `train_params` and `net_params` are defined in configuration file.
#### Training in mixed precision
Training in MP is slighty different, hence Pytorch `DataParallel`ization of model is not supported correctly. 
So we used `DistributedDataParallel` instead. You can launch training with following command
```
python -m torch.distributed.launch --nproc_per_node=$GPUS  main.py train --cuda --config PATH/TO/CONFIG
```
You typically want to lauch one process per gpu. To set mixed precision level, use option `amp_opt_level` in `train_params`.
## Evaluating trained model
```
./eval.py --dataset test-clean --checkpoint PATH/TO/CHECKPOINT --cuda
```
If you don't have GPU in your machine, you can evaluate on cpu by ommiting `--cuda`.
Using `--beam-width`, `--alpha`, `--beta`, `--lm-file`, `--trie-file` you can adjust beam search algorithm's parameters and provide custom language model. Please use
```
./eval.py --help
```
to see default values.
Besides calculating WER on Librispeech dataset, you can also try to generate transcription for your own audio files. To do that simply use
```
./eval.py --track-dir PATH/TO/FOLDER/WITH/YOUR/FILES --checkpoint PATH/TO/CHECKPOINT --cuda
```
## Results
The model converges after training for 11.5 hours in mixed precision and around 24 hours in full precision. Tested on DGX station equipped with 4xNvidia Tesla V100 GPUs. We trained on 960h Librispeech (train-clean + train-other) and tested on Librispeech test-clean.
We managed to achieve 10.37 WER while training in FP32 and 11.57 WER while training in mixed precision.


## Authors

* **Piotr Ambroszczyk** - [ambroszczyk](https://github.com/ambroszczyk)
* **Łukasz Kondraciuk** - [xman1024](https://github.com/xman1024)
* **Wojciech Przybyszewski** - [przybyszewskiw](https://github.com/przybyszewskiw)
* **Jan Tabaszewski** - [tabasz](https://github.com/tabasz)


## License

This project is licensed under the MIT License - see the [LICENCE.md](LICENCE.md) file for details.

## References
* [Deep Speech](https://arxiv.org/pdf/1412.5567.pdf)
* [Deep Speech 2](https://arxiv.org/pdf/1512.02595.pdf)
* [Librispeech](http://www.openslr.org/12)
