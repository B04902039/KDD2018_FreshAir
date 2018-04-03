import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_dim, args):
        super(EncoderRNN, self).__init__()
        self.input_dim = hidden_dim
        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout
        self.use_cuda = args.use_cuda
        self.use_bidirection = args.use_bidirection
        

        if self.use_bidirection:
            self.gru = nn.GRU(input_dim, hidden_dim//2, dropout = self.dropout, bidirectional=True)
        else :
            self.gru = nn.GRU(input_dim, hidden_dim, dropout = self.dropout)

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
            self.gru = nn.GRU(input_dim, hidden_dim//2, dropout = self.dropout, bidirectional=True)
        else :
            self.gru = nn.GRU(input_dim, hidden_dim, dropout = self.dropout)
        self.out = nn.Linear(hidden_dim, output_dim)

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
        output = self.out(output[0])
        return output, hidden
