import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Attn(nn.Module):
    def __init__(self, output_length, args):
        super(Attn, self).__init__()
        
        self.method = args.method
        self.hidden_dim = args.hidden_dim
        self.output_length = output_length
        self.dropout = args.dropout
        self.use_cuda = args.use_cuda

        if self.method == 'general':
            self.v = nn.Linear(self.hidden_dim, 1)
        if self.method == 'concat':
            self.attn = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            self.v = nn.Linear(self.hidden_dim, 1)

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        if self.method == 'dot':
            attn_weights = torch.bmm(encoder_outputs,
                hidden.unsqueeze(2)).squeeze(2)
        if self.method == 'general' or self.method == 'concat':
            hiddens = hidden.unsqueeze(1)
            hiddens = hiddens.repeat(1, max_len, 1)
            hiddens = torch.cat((hiddens, encoder_outputs), 2)
            attn_weights = Variable(torch.zeros(batch_size, max_len))
            if self.use_cuda:
                attn_weights = attn_weights.cuda()
            for i in range(max_len):
                attn_weights[:, i] = self.score(hiddens[:, i])
        return F.softmax(attn_weights, dim = 1)
    
    def score(self, hidden):
        if self.method == 'general':
            energy = self.v(hidden)
        if self.method == 'concat':
            energy = self.attn(hidden)
            energy = self.v(energy)
        return energy


class AttnDecoderRNN(nn.Module):
    def __init__(self, input_shape, output_shape, args):
        super(AttnDecoderRNN, self).__init__()
        self.input_length, self.input_dim = input_shape
        self.output_length, self.output_dim = output_shape
        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout
        self.use_cuda = args.use_cuda
        self.use_bidirection = args.use_bidirection
        
        self.attn = nn.Linear(self.hidden_dim + self.output_dim, self.input_length)
        self.attn_combine = nn.Linear(self.hidden_dim + self.output_dim, self.hidden_dim)
        if self.use_bidirection:
            self.gru = nn.GRU(self.hidden_dim, self.hidden_dim//2,
                dropout = self.dropout, bidirectional=True)
        else :
            self.gru = nn.GRU(self.hidden_dim, self.hidden_dim,
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

    def forward(self, input, last_hidden, encoder_outputs):
        input = input[0,:,:3]
        x = torch.cat((input, last_hidden[0]), 1)
        attn_weights = F.softmax(
            self.attn(x), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs.transpose(0, 1))
        context = context.squeeze(1)
        context = torch.cat((input, context), 1)
        context = self.attn_combine(context).unsqueeze(0)
        context = F.relu(context)
        rnn_out, hidden = self.gru(context, last_hidden)
        rnn_out = rnn_out.squeeze(0)
        hidden = hidden.squeeze(0)
        output = self.out(rnn_out)
        return output, hidden

class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, input_shape, output_shape, args):
        super(BahdanauAttnDecoderRNN, self).__init__()
        self.input_length, self.input_dim = input_shape
        self.output_length, self.output_dim = output_shape
        self.hidden_dim = args.hidden_dim
        self.attn_model = Attn(self.output_length, args)
        self.dropout = args.dropout
        self.use_cuda = args.use_cuda
        self.use_bidirection = args.use_bidirection
        
        if self.use_bidirection:
            self.gru = nn.GRU(self.hidden_dim + self.output_dim, 
                self.hidden_dim//2, dropout = self.dropout, bidirectional=True)
        else :
            self.gru = nn.GRU(self.hidden_dim + self.output_dim,
                self.hidden_dim, dropout = self.dropout)
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

    def forward(self, input, last_hidden, encoder_outputs):
        input = input[:,:,:3]
        attn_weights = self.attn_model(last_hidden[0], encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1),
                        encoder_outputs.transpose(0, 1))
        context = context.transpose(0, 1)
        context = torch.cat((input, context), 2)
        rnn_out, hidden = self.gru(context, last_hidden)
        rnn_out = rnn_out.squeeze(0)
        hidden = hidden.squeeze(0)
        output = self.out(rnn_out)
        return output, hidden

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, input_shape, output_shape, args):
        super(LuongAttnDecoderRNN, self).__init__()
        self.input_length, self.input_dim = input_shape
        self.output_length, self.output_dim = output_shape
        self.hidden_dim = args.hidden_dim
        self.attn_model = Attn(self.output_length, args)
        self.dropout = args.dropout
        self.use_cuda = args.use_cuda
        self.use_bidirection = args.use_bidirection
        
        if self.use_bidirection:
            self.gru = nn.GRU(self.output_dim, self.hidden_dim//2, dropout = self.dropout, bidirectional=True)
        else :
            self.gru = nn.GRU(self.output_dim, self.hidden_dim, dropout = self.dropout)

        self.concat = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
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

    def forward(self, input, last_hidden, encoder_outputs):
        input = input[:,:,:3]
        rnn_out, hidden = self.gru(input, last_hidden)
        attn_weights = self.attn_model(rnn_out[0], encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1),
                        encoder_outputs.transpose(0, 1))
        rnn_out = rnn_out.squeeze(0) 
        hidden = hidden.squeeze(0)
        context = context.squeeze(1)     
        context = torch.cat((rnn_out, context), 1)
        output = F.tanh(self.concat(context))
        output = self.out(rnn_out)
        return output, hidden