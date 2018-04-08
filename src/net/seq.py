import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_shape, args):
        super(EncoderRNN, self).__init__()
        self.input_length, self.input_dim,  = input_shape
        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout
        self.use_cuda = args.use_cuda
        self.use_bidirection = args.use_bidirection
        if self.use_bidirection:
            self.gru = nn.GRU(self.input_dim, self.hidden_dim//2, dropout = self.dropout, bidirectional=True)
        else :
            self.gru = nn.GRU(self.input_dim, self.hidden_dim, dropout = self.dropout)

    def init_hidden(self, batch_size):
        if self.use_bidirection:
            if self.use_cuda:
                h0 = Variable(torch.zeros(2, batch_size, self.hidden_dim // 2).cuda())
            else:
                h0 = Variable(torch.zeros(2, batch_size, self.hidden_dim // 2))
        else :
            if self.use_cuda:
                h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())
            else:
                h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        return h0

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, output_shape, args):
        super(DecoderRNN, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.output_length, self.output_dim,  = output_shape
        self.dropout = args.dropout
        self.use_cuda = args.use_cuda
        self.use_bidirection = args.use_bidirection
        
        if self.use_bidirection:
            self.gru = nn.GRU(self.output_dim, self.hidden_dim//2,
                dropout = self.dropout, bidirectional=True)
        else :
            self.gru = nn.GRU(self.output_dim, self.hidden_dim,
                dropout = self.dropout)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)

    def init_hidden(self, batch_size):
        if self.use_bidirection:
            if self.use_cuda:
                h0 = Variable(torch.zeros(2, batch_size, self.hidden_dim // 2).cuda())
            else:
                h0 = Variable(torch.zeros(2, batch_size, self.hidden_dim // 2))
        else :
            if self.use_cuda:
                h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())
            else:
                h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        return h0
        
    def forward(self, input, last_hidden, encoder_outputs = None):
        x = input[:,:,:3]
        output, hidden = self.gru(x, last_hidden)
        output = output.squeeze(0)
        hidden = hidden.squeeze(0)
        output = self.out(output)
        return output, hidden

class DecoderDNN(nn.Module):
    def __init__(self, output_shape, args):
        super(DecoderDNN, self).__init__()
        self.output_length, self.output_dim,  = output_shape
        self.hidden_dim = args.hidden_dim
        self.use_cuda = args.use_cuda
        self.out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, encoder_outputs):
        maxlen = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        output = Variable(torch.zeros(self.output_length, batch_size,  self.output_dim))
        encoder_outputs = encoder_outputs[-self.output_length:]
        if (self.use_cuda):
            output = output.cuda()
        for i in range(self.output_length):
            output[i, :,:] = self.out(encoder_outputs[i, :])
        return output