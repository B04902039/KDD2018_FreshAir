import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Attn(nn.Module):
    def __init__(self, args):
        super(Attn, self).__init__()
        
        self.method = args.method
        self.hidden_dim = args.hidden_dim
        self.encoder_length = args.encoder_length
        self.dropout = args.dropout
        self.use_cuda = args.use_cuda
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_dim, hidden_dim)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_dim * 2, hidden_dim)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_dim))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)

        attn_energies = Variable(torch.zeros(batch_size, max_len))

        if self.use_cuda:
            attn_energies = attn_energies.cuda()

        for b in range(batch_size):
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        return F.softmax(attn_energies).unsqueeze(1)
    
    def score(self, hidden, encoder_output):
        
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.dot(energy)
            return energy


class AttnDecoderRNN(nn.Module):
    def __init__(self, output_dim, args):
        super(AttnDecoderRNN, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = args.hidden_dim
        self.encoder_length = args.encoder_length
        self.dropout = args.dropout
        self.use_cuda = args.use_cuda
        self.use_bidirection = args.use_bidirection
        
        self.attn = nn.Linear(self.hidden_dim * 2, self.encoder_length)
        self.attn_combine = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        if self.use_bidirection:
            self.gru = nn.GRU(input_dim, hidden_dim//2, dropout = self.dropout, bidirectional=True)
        else :
            self.gru = nn.GRU(input_dim, hidden_dim, dropout = self.dropout)
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

    def forward(self, input, hidden, encoder_outputs):

        attn_weights = F.softmax(
            self.attn(torch.cat((input[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        output = torch.cat((input[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        return output, hidden, attn_weights

class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, output_dim, args):
        super(BahdanauAttnDecoderRNN, self).__init__()
        self.attn_model = Attn(args)
        self.output_dim = output_dim
        self.hidden_dim = args.hidden_dim
        self.encoder_length = args.encoder_length
        self.dropout = args.dropout
        self.use_cuda = args.use_cuda
        self.use_bidirection = args.use_bidirection
        
        self.attn = Attn('concat', hidden_size)
        if self.use_bidirection:
            self.gru = nn.GRU(input_dim, hidden_dim//2, dropout = self.dropout, bidirectional=True)
        else :
            self.gru = nn.GRU(input_dim, hidden_dim, dropout = self.dropout)
        self.out = nn.Linear(hidden_size, output_size)
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

    def forward(self, input, last_hidden, encoder_outputs, batch_size):
        
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        context = context.transpose(0, 1)
        rnn_input = torch.cat((input, context), 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)
        return output, hidden, attn_weights

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, output_dim, args):
        super(LuongAttnDecoderRNN, self).__init__()

        self.attn_model = Attn(args)
        self.output_dim = output_dim
        self.hidden_dim = args.hidden_dim
        self.encoder_length = args.encoder_length
        self.dropout = args.dropout
        self.use_cuda = args.use_cuda
        self.use_bidirection = args.use_bidirection
        
        if self.use_bidirection:
            self.gru = nn.GRU(input_dim, hidden_dim//2, dropout = self.dropout, bidirectional=True)
        else :
            self.gru = nn.GRU(input_dim, hidden_dim, dropout = self.dropout)

        self.concat = nn.Linear(2* hidden_dim, hidden_dim)
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

    def forward(self, input, last_hidden, encoder_outputs, batch_size):

        rnn_out, hidden = self.gru(input, last_hidden)
        attn_weights = self.attn(rnn_out, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        rnn_out = rnn_out.squeeze(0) 
        context = context.squeeze(1)     
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        output = self.out(concat_output)

        return output, hidden, attn_weights
