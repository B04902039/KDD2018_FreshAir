import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_dim, args):
        super(EncoderRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout
        self.use_cuda = args.use_cuda
        self.use_bidirection = args.use_bidirection
        if self.use_bidirection:
            self.gru = nn.GRU(input_dim, self.hidden_dim//2, dropout = self.dropout, bidirectional=True)
        else :
            self.gru = nn.GRU(input_dim, self.hidden_dim, dropout = self.dropout)

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
    def __init__(self, output_dim, args):
        super(DecoderRNN, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = args.hidden_dim
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
