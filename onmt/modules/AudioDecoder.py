import math
import torch.nn as nn
import torch.nn.functional as F

class AudioDecoder(nn.Module):
    """
    A simple encoder convolutional -> recurrent neural network for
    audio input.

    Args:
        num_layers (int): number of encoder layers.
        bidirectional (bool): bidirectional encoder.
        rnn_size (int): size of hidden states of the rnn.
        dropout (float): dropout probablity.
        sample_rate (float): input spec
        window_size (int): input spec

    """
    def __init__(self, num_layers, bidirectional, rnn_size, dropout,
                 downsample, window_size):
        super(AudioEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = rnn_size
        self.downsample = 8
        self.n_feats = 41
        
        # input_size = 608
        self.rnn = nn.LSTM(rnn_size, rnn_size,
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional)
        self.linear = nn.Linear(rnn_size, self.downsample*self.n_feats)

    def forward(self, input, context, state, context_lengths=None):
        "See :obj:`onmt.modules.EncoderBase.forward()`"

        output, hidden = self.rnn(context)
        output = self.linear(output)
        output = output.reshape(output.size(0), -1, self.n_feats)

        return output
