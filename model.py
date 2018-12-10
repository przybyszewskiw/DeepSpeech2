from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class Convolutions(nn.Module):
    """
    First part of DeepSpeech net. Consists of one or more 1d convolutions.

    Constructor parameters:
        convolutions = number of convolutional layers
        frequencies = number of different frequencies per time stamp
        context = number of neighbouring time stamps we should care about
        (implies kernel size)

    Input: Tensor of shape NxFxT where N is batch size, F is the number of
           different frequencies and T is lenght of time-series.
    Output: Tensor of the same shape as input
    """
    def __init__(self, conv_number=2, frequencies=700, context=5):
        super(Convolutions, self).__init__()
        self.frequencies = frequencies
        self.conv_number = conv_number
        self.context = context

        self.layers = []
        # TODO Is that what we really want? (namely are those the convolutions
        # over the time dimension that the paper tells us about)
        for _ in range(self.conv_number):
            new_layer = nn.Sequential(
              nn.Conv1d(in_channels=self.frequencies, out_channels=self.frequencies,
                        kernel_size=2*self.context+1, padding=self.context,
                        groups=self.frequencies),
              nn.Hardtanh(min_val=0, max_val=20, inplace=True)
            )
            self.layers.append(new_layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Recurrent(nn.Module):
    """
    Second part of DeepSpeech. Consists of one or more recurrent layers.

    Constructor parameters:
        rec_number = number of recurrent layers
        frequencies = number of frequencies per one time stamp

    Input: Tensor of shape NxFxT where N is batch size, F is the number of
           different frequencies and T is lenght of time-series.
    Output: Tensor of the shape NxTxF where B, T and F as in input
    """
    def __init__(self, rec_number=3, frequencies=700):
        super(Recurrent, self).__init__()
        self.frequencies = frequencies
        self.rec_number = rec_number
        # TODO Use Hardtanh(0, 20) from paper instead of tanh or simple ReLU
        # which are default for torch.nn.RNN
        self.layers = []
        for _ in range(self.rec_number):
            new_layer = nn.RNN(input_size=self.frequencies, hidden_size=self.frequencies,
                               bidirectional=True)
            self.layers.append(new_layer)

    def forward(self, x):
        x = x.transpose(1, 2)
        for layer in self.layers:
            x, _ = layer(x)
            (x1, x2) = torch.chunk(x, 2, dim=2)
            x = x1 + x2
        return x

class FullyConnected(nn.Module):
    """
    Third part of DeepSpeech. Consists of one or more fully connected layers.

    Constructor parameters:
        ful_number = number of fully connected layers
        frequencies = number of frequencies per one time stamp

    Input: Tensor of shape NxTxF where N is batch size, F is the number of
           different frequencies and T is lenght of time-series.
    Output: Tensor of the same shape as input
    """
    def __init__(self, full_number=2, frequencies=700):
        super(FullyConnected, self).__init__()
        self.full_number = full_number
        self.frequencies = frequencies
        self.layers = []
        for _ in range(self.full_number):
            new_layer = nn.Sequential(
              nn.Linear(self.frequencies, self.frequencies),
              nn.Hardtanh(min_val=0, max_val=20, inplace=True)
            )
            self.layers.append(new_layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Probabilities(nn.Module):
    """
    Fourth part of DeepSpeech. Consist of one fully connected layer which defines
    what is probability for each character in a given time moment and softmax
    layer normalizing those probabilities using LogSoftmax (because PyTorch
    CTCLoss requires LogSoftmax instead of 'standard' Softmax)

    Constructor parameters:
        characters = number of characters which we are predicting
        frequencies = number of frequencies per one time stamp

    Input: Tensor of shape NxTxF where N is batch size, F is the number of
           different frequencies and T is lenght of time-series.
    Output: Tensor of shape NxTxC where N and T are the same as in the input and
            C is the number of character which we make predictions about.
    """
    def __init__(self, characters=29, frequencies=700):
        super(Probabilities, self).__init__()
        self.characters = characters
        self.frequencies = frequencies
        self.layer = nn.Sequential(
          nn.Linear(frequencies, characters),
          nn.LogSoftmax(dim=2)
        )

    def forward(self, x):
        return self.layer(x)

#class DeepSpeech(nn.Module):

def test():
    x = torch.rand(1, 30, 5)
    print(x)
    net1 = Convolutions(frequencies=30, context=1, conv_number=1)
    x = net1(x)
    print(x)
    net2 = Recurrent(frequencies=30)
    x = net2(x)
    print(x)
    net3 = FullyConnected(frequencies=30)
    x = net3(x)
    print(x)
    net4 = Probabilities(frequencies=30)
    x = net4(x)
    print(x)
    print(x.shape)

if __name__ == "__main__":
    test()
