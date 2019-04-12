from __future__ import print_function

import torch
import torch.nn as nn


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
    def __init__(self, conv_number=2, frequencies=700, context=5, batch_norm=False):
        super(Convolutions, self).__init__()
        self.frequencies = frequencies
        self.conv_number = conv_number
        self.context = context

        self.layers = nn.ModuleList()
        # TODO Is that what we really want? (namely are those the convolutions
        # over the time dimension that the paper tells us about)

        for _ in range(self.conv_number):
            new_layer = nn.Sequential(
                nn.Conv1d(in_channels=self.frequencies, out_channels=self.frequencies,
                        kernel_size=2*self.context+1, padding=self.context,
                        groups=self.frequencies),
              nn.Hardtanh(min_val=0, max_val=20, inplace=True)
            )
            if batch_norm:
                new_layer = nn.Sequential(new_layer, nn.BatchNorm1d(self.frequencies, momentum=0.95,
          eps=1e-4))
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
           different frequencies and T is length of time-series.
    Output: Tensor of the shape NxTxF where N, T and F as in input
    """
    def __init__(self, rec_number=3, frequencies=700):
        super(Recurrent, self).__init__()
        self.frequencies = frequencies
        self.rec_number = rec_number
        # TODO Use Hardtanh(0, 20) from paper instead of tanh or simple ReLU
        # which are default for torch.nn.RNN
        self.layers = nn.ModuleList()
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
        full_number = number of fully connected layers
        frequencies = number of frequencies per one time stamp

    Input: Tensor of shape NxTxF where N is batch size, F is the number of
           different frequencies and T is lenght of time-series.
    Output: Tensor of the same shape as input
    """
    def __init__(self, full_number=2, frequencies=700, dropout=0):
        super(FullyConnected, self).__init__()
        self.full_number = full_number
        self.frequencies = frequencies
        self.layers = nn.ModuleList()
        for _ in range(self.full_number):
            new_layer = nn.Sequential(
              nn.Linear(self.frequencies, self.frequencies),
              nn.Hardtanh(min_val=0, max_val=20, inplace=True)
            )
            if dropout != 0:
                new_layer = nn.Sequential(new_layer, nn.Dropout(dropout))
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
        # self.layer = nn.Sequential(
        #  nn.Linear(frequencies, characters),
        #  nn.LogSoftmax(dim=2)
        # )
        self.linear = nn.Linear(frequencies, characters)
        self.logsoft = nn.LogSoftmax(dim=2)
        self.soft = nn.Softmax(dim=2)

    def forward(self, x):
        # return self.layer(x)
        x = self.linear(x)
        return self.logsoft(x), self.soft(x)


class DeepSpeech(nn.Module):
    """
        Composition of the components: Convolutions, Recurrent, FullyConnected, Probabilities
        into one DeepSpeech net.

        Constructor parameters:
            frequencies = number of frequencies per one time stamp
            convolutions = number of convolutional layers
            context = number of neighbouring time stamps we should care about
            (implies kernel size)
            rec_number = number of recurrent layers
            full_number = number of fully connected layers
            characters = number of characters which we are predicting
            batch_norm = whether to use batch normalization before RNN

        Input: Tensor of shape NxFxT where N is batch size, F is the number of
           different frequencies and T is length of time-series.
        Output: Tensor of shape NxTxC where N and T are the same as in the input and
            C is the number of character which we make predictions about.

    """
    def __init__(self, frequencies=700, conv_number=2, context=5,
                 rec_number=3, full_number=2, characters=29, batch_norm=False, fc_dropout=0):
        super(DeepSpeech, self).__init__()
        self.characters = characters
        # TODO discuss whether to keep layer parameters (such as full_number) as the instance attributes
        self.full_number = full_number
        self.rec_number = rec_number
        self.context = context
        self.conv_number = conv_number
        self.frequencies = frequencies

        self.layer = nn.Sequential(
           Convolutions(conv_number=self.conv_number, frequencies=self.frequencies,
                         context=self.context, batch_norm=batch_norm),
           Recurrent(rec_number=self.rec_number, frequencies=self.frequencies),
           FullyConnected(full_number=self.full_number, frequencies=self.frequencies, dropout=fc_dropout),
           Probabilities(characters=self.characters, frequencies=self.frequencies)
        )
        print(self.layer)


    def forward(self, x):
        x, y = self.layer(x)
        return x, y

    """
        Calculate loss using CTC loss function.

        Input:
            output = Tensor of shape NxTxC where N and T are the same as in the input and
            C is the number of character which we make predictions about.
            target = tensor of shape NxS where N is batch size and S in length of
            utterances.

            constraints:
             -S <= T
             -target has to be a positive integer tensor #https://discuss.pytorch.org/t/ctcloss-dont-work-in-pytorch/13859/3
        Output: loss tensor
    """
    #TODO move to GPU (works only on CPU)
    @staticmethod
    def criterion(output, target, target_length=None):
        ctc_loss = nn.CTCLoss(reduction='mean')
        batch_size = output.shape[0]
        utterance_length = output.shape[1]
        output = output.transpose(0, 1)
        output_length = torch.full((batch_size,), utterance_length, dtype=torch.int32)

        if target_length is None:
            target_length = torch.full((target.shape[0],), target.shape[1])

        return ctc_loss(output, target, output_length, target_length)


def test():
    N = 1
    F = 30
    T = 5
    C = 10
    for _ in range(100):
        x = torch.rand(N, F, T)

        net = DeepSpeech(frequencies=F, context=1, conv_number=1, characters=C)
        x = net(x)

        labels = torch.randint(1, C, (N, T))
        loss = DeepSpeech.criterion(x, labels)

        if loss.item() == float('inf'):
            print("Bad:" + str(labels))
        else:
            print("Ok:" + str(labels))


if __name__ == "__main__":
    test()

