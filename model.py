from __future__ import print_function

import torch
import torch.nn as nn
import math


class Convolutions(nn.Module):
    """
    First part of DeepSpeech net. Consists of one or more 2d convolutions.

    Constructor parameters:
        frequencies = number of frequencies in input tensor
        conv_list = list of dicts des
        initial_channels = number of channels in input tensor
        batch_norm = whether or not use batch normalization after each convolution
    Input: Tensor of shape NxCxFxT where N is batch size, F is the number of
           different frequencies and T is lenght of time-series.
    Output: Tensor of the shape NxC'xF'xT'. Parameters C' and F' can be found as
            self.newC and self.newF respectively
    """

    def __init__(self, conv_layers, frequencies=160, initial_channels=1, conv_list=None, batch_norm=False):
        super(Convolutions, self).__init__()
        self.frequencies = frequencies
        self.conv_layers = conv_layers
        self.layers = nn.ModuleList()
        self.newC = initial_channels
        self.newF = frequencies

        for layer in self.conv_layers:
            new_layer = nn.Sequential(
                Conv2dSame(in_channels=self.newC, out_channels=layer["num_chan"],
                           kernel_size=layer["kernel"], stride=layer["stride"]),
                nn.Hardtanh(min_val=0, max_val=20, inplace=True)
            )

            if batch_norm:
                new_layer = nn.Sequential(new_layer,
                                          nn.BatchNorm2d(layer["num_chan"], momentum=0.95,
                                                         eps=1e-4))

            self.newC = layer["num_chan"]
            self.newF = math.ceil(float(self.newF) / float(layer["stride"][0]))
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

    def __init__(self, frequencies, rec_number=3):
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

    def __init__(self, frequencies, layers_sizes=[2048], dropout=0):
        super(FullyConnected, self).__init__()
        self.layers_sizes = list(zip([frequencies] + layers_sizes, layers_sizes))
        self.frequencies = frequencies
        self.layers = nn.ModuleList()
        for inner, outer in self.layers_sizes:
            new_layer = nn.Sequential(
                nn.Linear(inner, outer),
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

    def __init__(self, frequencies, characters=29):
        super(Probabilities, self).__init__()
        self.characters = characters
        self.frequencies = frequencies
        self.linear = nn.Linear(frequencies, characters)
        self.logsoft = nn.LogSoftmax(dim=2)
        self.soft = nn.Softmax(dim=2)

    def forward(self, x):
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
            initializer = function to initialize weights with

        Input: Tensor of shape NxFxT where N is batch size, F is the number of
           different frequencies and T is length of time-series.
        Output: Tensor of shape NxTxC where N and T are the same as in the input and
            C is the number of character which we make predictions about.

    """

    def __init__(self, conv_layers,
                 conv_initial_channels=160,
                 rec_number=3,
                 fc_layers_sizes=[2048],
                 characters=29,
                 batch_norm=False,
                 fc_dropout=0,
                 initializer=None):
        super(DeepSpeech, self).__init__()
        self.characters = characters
        # TODO discuss whether to keep layer parameters (such as full_number) as the instance attributes
        self.full_layers = fc_layers_sizes
        self.rec_number = rec_number
        self.frequencies = conv_initial_channels
        self.conv_layers = conv_layers
        self.initializer = initializer

        self.convs = Convolutions(frequencies=self.frequencies, conv_layers=self.conv_layers, batch_norm=batch_norm)
        self.rec = Recurrent(rec_number=self.rec_number,
                             frequencies=self.convs.newF * self.convs.newC)
        self.fc = FullyConnected(layers_sizes=self.full_layers,
                                 frequencies=self.convs.newF * self.convs.newC, dropout=fc_dropout)
        self.probs = Probabilities(characters=self.characters, frequencies=fc_layers_sizes[-1])
        if initializer:
            self.apply(self.weights_init)

        print("Net structure:", self.convs, self.rec, self.fc, self.probs)

    def weights_init(self, m):
        if hasattr(m, "weight") and m.weight.dim() > 1:
            self.initializer(m.weight)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.convs(x)
        (n, c, f, t) = x.shape
        x = x.view(n, c * f, t)
        x = self.rec(x)
        x = self.fc(x)
        return self.probs(x)

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

    # TODO move to GPU (works only on CPU)
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

        net = DeepSpeech(conv_initial_channels=F, context=1, conv_number=1, characters=C)
        x, _ = net(x)

        labels = torch.randint(1, C, (N, T))
        loss = DeepSpeech.criterion(x, labels)

        if loss.item() == float('inf'):
            print("Bad:" + str(labels))
        else:
            print("Ok:" + str(labels))


class Conv2dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), dilation=(1, 1)):
        super(Conv2dSame, self).__init__()
        self.F = kernel_size
        self.S = stride
        self.D = dilation
        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation=dilation)

    def forward(self, x_in):
        n, c, h, w = x_in.shape
        h2 = math.ceil(h / self.S[0])
        w2 = math.ceil(w / self.S[1])
        pr = max(0, (h2 - 1) * self.S[0] + (self.F[0] - 1) * self.D[0] + 1 - h)
        pc = max(0, (w2 - 1) * self.S[1] + (self.F[1] - 1) * self.D[1] + 1 - w)
        x_pad = nn.ZeroPad2d((pc // 2, pc - pc // 2, pr // 2, pr - pr // 2))(x_in)
        x_out = self.layer(x_pad)
        return x_out


if __name__ == "__main__":
    test()

