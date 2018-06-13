import math
import torch.nn as nn
import torch.nn.functional as F

class pBLSTMLayer(nn.Module):
    def __init__(self,input_feature_dim,hidden_dim,rnn_unit='LSTM',dropout_rate=0.0, downsample=2, bi=True):
        super(pBLSTMLayer, self).__init__()
        self.rnn_unit = getattr(nn,rnn_unit.upper())
        self.downsample = downsample
        self.bi = bi
        # feature dimension will be doubled since time resolution reduction
        self.BLSTM = self.rnn_unit(input_feature_dim*self.downsample,hidden_dim,1, bidirectional=self.bi,
            dropout=dropout_rate,batch_first=True)

    def forward(self,input_x):
        batch_size = input_x.size(0)
        timestep = input_x.size(1)
        feature_dim = input_x.size(2)
        # Reduce time resolution
        input_x = input_x.contiguous().view(batch_size, int(timestep/self.downsample),feature_dim*self.downsample)
        # Bidirectional RNN
        output,hidden = self.BLSTM(input_x)
        return output,hidden

class AudioEncoder(nn.Module):
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
                 sample_rate, window_size):
        super(AudioEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = rnn_size

        '''
        self.layer1 = nn.Conv2d(1,   32, kernel_size=(11, 11),
                                padding=(0, 10), stride=(2, 2))
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(32,  32, kernel_size=(21, 11),
                                padding=(0, 0), stride=(2, 1))
        self.batch_norm2 = nn.BatchNorm2d(32)
        '''
        #nFeats = sample_rate * window_size
        '''
        nFeats = 120
        input_size = int(math.floor((nFeats) / 2) + 1)
        input_size = int(math.floor(input_size - 11) / 2 + 1)
        input_size = int(math.floor(input_size - 21) / 2 + 1)
        input_size *= 32
        '''
        # input_size = 608
        input_size = 41
        self.rnn = nn.LSTM(input_size, rnn_size,
                           num_layers=1, #num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional)
        self.pLSTM_layer1 = pBLSTMLayer(rnn_size*2,rnn_size, rnn_unit='lstm', dropout_rate=dropout)
        self.pLSTM_layer2 = pBLSTMLayer(rnn_size*2,rnn_size, rnn_unit='lstm', dropout_rate=dropout)
        self.pLSTM_layer3 = pBLSTMLayer(rnn_size*2,rnn_size, rnn_unit='lstm', dropout_rate=dropout)

    def load_pretrained_vectors(self, opt):
        # Pass in needed options only when modify function definition.
        pass

    def forward(self, input, lengths=None):
        "See :obj:`onmt.modules.EncoderBase.forward()`"

        '''
        # (batch_size, 1, nfft, t)
        # layer 1
        input = self.batch_norm1(self.layer1(input[:, :, :, :]))

        # (batch_size, 32, nfft/2, t/2)
        input = F.hardtanh(input, 0, 20, inplace=True)

        # (batch_size, 32, nfft/2/2, t/2)
        # layer 2
        input = self.batch_norm2(self.layer2(input))

        # (batch_size, 32, nfft/2/2, t/2)
        input = F.hardtanh(input, 0, 20, inplace=True)

        '''
        #print "original input shape:", input.size()
        input = input.squeeze()
        #print "squeezed input shape:", input.size()
        #batch_size = input.size(0)
        #length = input.size(2)
        #input = input.view(batch_size, -1, length)
        try:
            input = input.transpose(0, 2).transpose(1, 2)
        except:
            input = input.view(1, input.size(0), input.size(1))
            input = input.transpose(0, 2).transpose(1, 2)
        #print "transposed input shape:", input.size()
        
        #print "actual input shape:", input.size()
        output, hidden = self.rnn(input)
        #print "into 1:", output.size()
        #print "rnn hidden:", hidden[0].size()
        output = output.transpose(0, 1)
        #print "into 1:", output.size()
        output, _ = self.pLSTM_layer1(output)
        #print "into 2:", output.size()
        output, _ = self.pLSTM_layer2(output)
        #print "into 3:", output.size()
        output, hidden = self.pLSTM_layer3(output)
        #print "output:", output.size()
        #print "hidden:", len(hidden), hidden[0].size()
        output = output.transpose(0, 1)
        #print "output:", output.size()
        
        return hidden, output

class GlobalAudioEncoder(nn.Module):
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
                 sample_rate, window_size):
        super(GlobalAudioEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = rnn_size

        input_size = 41
        self.rnn = nn.LSTM(input_size, rnn_size,
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=False)

    def load_pretrained_vectors(self, opt):
        # Pass in needed options only when modify function definition.
        pass

    def forward(self, input, lengths=None):
        input = input.squeeze()
        print "original input shape:", input.size()
        try:
            input = input.transpose(0, 2).transpose(1, 2)
        except:
            input = input.view(1, input.size(0), input.size(1))
            input = input.transpose(0, 2).transpose(1, 2)
        print "transposed input shape:", input.size()
        
        #print "actual input shape:", input.size()
        output, hidden = self.rnn(input)

        print "output shape:", output[-1, :, :].size()
        
        return hidden, output[-1, :, :]

class ConvGlobalAudioEncoder(nn.Module):
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
                 sample_rate, window_size):
        super(ConvGlobalAudioEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = rnn_size

        '''
        self.layer1 = nn.Conv2d(1,   32, kernel_size=(11, 11),
                                padding=(0, 10), stride=(2, 2))
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(32,  32, kernel_size=(21, 11),
                                padding=(0, 0), stride=(2, 1))
        self.batch_norm2 = nn.BatchNorm2d(32)
        '''

        input_size = 41
        # input = batch x 1 x 41 x len
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1, 32, kernel_size=(input_size-5, 1), stride=1), # out = batch x 32 x 5 x len
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.MaxPool2d((5,3)))
        # output = batch x 32 x 1 x len / 3
        self.layer2 = nn.Sequential(
                        nn.Conv2d(32, 64, kernel_size=(1, 5), stride=1), # out = batch x 64 x 1 x len / 5
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d((1,5)))
        # out = batch x 64 x 1 x len / 15
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64, self.hidden_size, kernel_size=(1, 3), stride=1), # out = batch x hidden x 1 x len / 15
                        nn.BatchNorm2d(self.hidden_size),
                        nn.ReLU())
        
    def load_pretrained_vectors(self, opt):
        # Pass in needed options only when modify function definition.
        pass

    def forward(self, input, lengths=None):
        input = input.squeeze()
        print "original input shape:", input.size()
        try:
            input = input.contiguous().view(input.size(0), 1, input.size(1), input.size(2))
        except:
            input = input.contiguous().view(1, 1, input.size(0), input.size(1))
        print "modified input shape:", input.size()

        # should be batch x channel x height x width
        #   = batch x 1 x 41 x len
        output = self.layer1(input)

        # output = batch x 32 x 1 x len / 3
        print "output shape 1:", output.size()

        output = self.layer2(output)

        # out = batch x 64 x 1 x len / 15
        print "output shape 2:", output.size()

        output = self.layer3(output)
        # out = batch x hidden x 1 x 1
        print "output shape 3:", output.size()

        output = output.max(dim=3)[0]
        hidden_output = output.contiguous().view(1, output.size(0), output.size(1))
        return (hidden_output, hidden_output), output.squeeze()


