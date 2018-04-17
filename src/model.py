import numpy as np
from random import random  
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from net import *
from evaluate import *

class AirNet(object):
    def __init__(self, input_shape, output_shape, args):
        self.use_cuda = args.use_cuda
        self.model_dir = args.model_dir
        self.meta_dir = args.meta_dir

        self.epochs = args.epochs
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.dropout = args.dropout

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_length, self.input_dim = input_shape
        self.output_length, self.output_dim = output_shape

        self.hidden_dim = args.hidden_dim
        self.decoder_type = args. decoder_type
        
        self.lr = args.lr
        self.tfr = args.tfr

        self.criterion_type = args.criterion_type
        self.decoder_type = args.decoder_type
        self.optimizer_type = args.optimizer_type
        
        self.init_model(args)
        self.init_opt(args)
        self.init_criterion(args)

        if args.use_cuda:  
            self.encoder.cuda()
            self.decoder.cuda()
        self.tr_hist = []
        self.val_hist = []
        
    def init_model(self, args):
        self.encoder  = EncoderRNN(self.input_shape, args)
        if (args.decoder_type == 'attn'):
            self.decoder  = AttnDecoderRNN(self.input_shape, self.output_shape, args)
        elif (args.decoder_type == 'bahdan'):
            self.decoder  = BahdanauAttnDecoderRNN(self.input_shape, self.output_shape, args)
        elif (args.decoder_type == 'luong'):
            self.decoder  = LuongAttnDecoderRNN(self.input_shape, self.output_shape, args)
        elif (args.decoder_type == 'dnn'):
            self.decoder  = DecoderDNN(self.output_shape, args)
        else :
            self.decoder  = DecoderRNN(self.output_shape, args)

    def init_opt(self, args):
        if (args.optimizer_type == 'adam'):
            self.encoder_opt = optim.Adam(self.encoder.parameters(), lr=args.lr)
            self.decoder_opt = optim.Adam(self.decoder.parameters(), lr=args.lr)
        elif (args.optimizer_type == 'rmsprop'):
            self.encoder_opt = optim.RMSprop(self.encoder.parameters(), lr=args.lr)
            self.decoder_opt = optim.RMSprop(self.decoder.parameters(), lr=args.lr)
        else :
            self.encoder_opt = optim.SGD(self.encoder.parameters(), lr=args.lr)
            self.decoder_opt = optim.SGD(self.decoder.parameters(), lr=args.lr)

    def init_criterion(self, args):
        if (args.criterion_type == 'l1'):
            self.criterion = nn.MSELoss()
        else :
            self.criterion = nn.L1Loss()

    def run_batch(self, batch, tfr = 0.8, isTrain = True):
        batch_size = len(batch['input'])
        input_seq = batch['input'].transpose(0, 1)
        input_seq = Variable(input_seq)
        if isTrain:
            output_seq = batch['output'].transpose(0, 1)
            output_seq = Variable(output_seq)
        pred_seq = Variable(torch.zeros(self.output_length, batch_size, self.output_dim ))
        if self.use_cuda:
            input_seq = input_seq.cuda()
            pred_seq = pred_seq.cuda()
            if isTrain:
                output_seq = output_seq.cuda()

        encoder_hidden = self.encoder.init_hidden(batch_size)
        encoder_outputs, encoder_hidden = self.encoder(input_seq, encoder_hidden)

        if self.decoder_type == 'dnn':
            pred_seq = self.decoder(encoder_outputs)
        else :
            decoder_input = input_seq[-1]
            decoder_hidden = encoder_hidden[-1]
            for t in range(self.output_length):
                decoder_input = decoder_input.unsqueeze(0)
                decoder_hidden = decoder_hidden.unsqueeze(0)
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                pred_seq[t] = decoder_output
                if isTrain and random() < tfr:
                    decoder_input = output_seq[t]
                else :
                    decoder_input = decoder_output
                    
        if isTrain:
            loss = self.criterion(pred_seq, output_seq)
          #  copy_seq = input_seq[24:,:,:3]
          # copy_seq = input_seq[24:,:,:3].repeat(2,1,1)
          #  pred_seq = pred_seq + copy_seq
          #  output_seq = output_seq + copy_seq
            return loss, output_seq.transpose(0, 1), pred_seq.transpose(0, 1)

        else :
            return pred_seq.transpose(0, 1)
    
    def log(self, gts, preds, losss):
        _smape = smape(gts, preds)
        _mape = mape(gts, preds)
        _loss = np.mean(losss)
        print('loss : {:.3f} smape : {:.3f} mape : {:.3f}'.format(
            _loss, np.mean(_smape), _mape))
        return _loss, np.mean(_smape), _mape, _smape 

    def train(self, train_loader, valid_loader):
        self.steps = 0
        for e in range(self.epochs):
            GTs = []
            LOSSs = []
            PREDs = []
            for i, x in enumerate(train_loader):
                self.encoder_opt.zero_grad()
                self.decoder_opt.zero_grad()
                loss, gt, pred = self.run_batch(x, self.tfr)
                loss.backward()
                self.encoder_opt.step()
                self.decoder_opt.step()
                _loss = loss.data.cpu().numpy()
                _pred = pred.data.cpu().numpy()
                _gt = gt.data.cpu().numpy()
                if len(LOSSs) == 0 :
                    LOSSs = _loss
                    PREDs = _pred
                    GTs = _gt
                else :
                    LOSSs = np.concatenate([LOSSs, _loss], axis=0)
                    PREDs = np.concatenate([PREDs, _pred], axis=0)
                    GTs = np.concatenate([GTs, _gt], axis=0)
                self.steps += 1                
                if self.steps % self.save_interval == 0:
                    self.save(self.steps)
            if e % self.log_interval == 0:
                print('======== eval {} ========'.format(e))  
                print('======== train ========')
                _loss, _smape, _mape, smape_c  = self.log(GTs, PREDs, LOSSs)
                self.tr_hist.append([_loss, _smape, _mape, *smape_c])
                print('======== valid ========')
                _loss, _smape, _mape, smape_c = self.eval(valid_loader)
                self.val_hist.append([_loss, _smape, _mape, *smape_c])
        return self.tr_hist, self.val_hist

    def eval(self, valid_loader):  
        self.encoder.eval()
        self.decoder.eval()
        PREDs = []
        GTs = []
        LOSSs = []
        for i, x in enumerate(valid_loader):
            loss, gt, pred = self.run_batch(x, 0)
            _loss = loss.data.cpu().numpy()
            _pred = pred.data.cpu().numpy()
            _gt = gt.data.cpu().numpy()
            if len(LOSSs) == 0 :
                LOSSs = _loss
                PREDs = _pred
                GTs = _gt
            else :
                LOSSs = np.concatenate([LOSSs, _loss], axis=0)
                PREDs = np.concatenate([PREDs, _pred], axis=0)
                GTs = np.concatenate([GTs, _gt], axis=0)
        _loss, _smape, _mape, smape_c  = self.log(GTs, PREDs, LOSSs)
        self.encoder.train()
        self.decoder.train()
        return _loss, _smape, _mape, smape_c
        
    def test(self, test_loader):  
        self.encoder.eval()
        self.decoder.eval()
        PREDs = []
        for i, x in enumerate(test_loader):
            pred = self.run_batch(x, 0, False)
            _pred = pred.cpu().data.numpy()
            PREDs.append(_pred)
        return np.asarray(PREDs)

    def adjust_lr(self, iter):
        lr = self.lr * (0.8 ** (iter/5000))
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr
        
    def save(self, iters):
        torch.save(self.encoder.state_dict(), '{}/{}_{}_encoder.pkl'.format(
            self.model_dir, iters, self.decoder_type))             
        torch.save(self.decoder.state_dict(), '{}/{}_{}_decoder.pkl'.format(
            self.model_dir, iters, self.decoder_type))  

    def load(self, encoder_path, decoder_path):
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=lambda storage, loc: storage))
