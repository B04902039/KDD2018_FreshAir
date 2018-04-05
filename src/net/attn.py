import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Attn(nn.Module):
    def __init__(self, args):
        super(Attn, self).__init__()
        
        self.method = args.method
        self.hidden_dim = args.hidden_dim
        self.input_length = args.input_length
        self.dropout = args.dropout
        self.use_cuda = args.use_cuda
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_dim, self.hidden_dim)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            self.v = nn.Parameter(torch.FloatTensor(1, self.hidden_dim))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)

        attn_weights = Variable(torch.zeros(batch_size, max_len))

        if self.use_cuda:
            attn_weights = attn_weights.cuda()

        for b in range(batch_size):
            for i in range(max_len):
                attn_weights[b, i] = self.score(hidden[b, :].unsqueeze(0),
                    encoder_outputs[i, b].unsqueeze(0))

        return F.softmax(attn_weights, dim = 1)
    
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
    def __init__(self, input_dim, output_dim, args):
        super(AttnDecoderRNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = args.hidden_dim
        self.input_length = args.input_length
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
    def __init__(self, input_dim, output_dim, args):
        super(BahdanauAttnDecoderRNN, self).__init__()
        self.attn_model = Attn(args)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = args.hidden_dim
        self.input_length = args.input_length
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
    def __init__(self, input_dim, output_dim, args):
        super(LuongAttnDecoderRNN, self).__init__()

        self.attn_model = Attn(args)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = args.hidden_dim
        self.input_length = args.input_length
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