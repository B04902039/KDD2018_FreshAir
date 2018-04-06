
# -*- coding: utf-8 -
import argparse
def parse():
    parser = argparse.ArgumentParser(description='kdd2018 pm2.5')
    # flag
    parser.add_argument('--test', action='store_true', help='whether testing')
    parser.add_argument('--train', action='store_true', help='whether training')
    parser.add_argument('--use-cuda', action='store_true', help='whether use gpu')
    parser.add_argument('--use-bidirection', action='store_true', help='whether use bidirection')
    parser.add_argument('--load', action='store_true', help='whether load model')
    # directory
    parser.add_argument('--meta-dir', type=str, default='meta', help='meta dir')        
    parser.add_argument('--model-dir', type=str, default='model', help='model dir')        
    parser.add_argument('--data-dir', type=str, default='data', help='data dir')        
    # path
    parser.add_argument('--train-path', type=str, default='data/train_fillna.csv', help='train file') 
    parser.add_argument('--test-path', type=str, default='data/test.csv', help='test file')   
    # model type
    parser.add_argument('--decoder-type', type=str, default='seq', help='model type')        
    parser.add_argument('--optimizer-type', type=str, default='rmsprop', help='model opt')        
    parser.add_argument('--criterion-type', type=str, default='mse', help='model loss')    
    parser.add_argument('--method', type=str, default='concat', help='attn method')        
    # model structure
    parser.add_argument('--input-length', type=int, default=48, help='input length')        
    parser.add_argument('--output-length', type=int, default=48, help='output length')     
    parser.add_argument('--hidden-dim', type=int, default=10, help='hidden dim')   
    # train parameter
    parser.add_argument('--valid-ratio', type=float, default=0.2, help='valid ratio')        
    parser.add_argument('--tfr', type=float, default=0.8, help='teacher forcing ratio')  
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')   
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')   
    parser.add_argument('--epochs', type=int, default=100, help='epochs')   
    parser.add_argument('--batch-size', type=int, default=8, help='epochs')   
    # interval 
    parser.add_argument('--log-interval', type=int, default=20, help='log interval')     
    parser.add_argument('--save-interval', type=int, default=100, help='save interval')      

    args = parser.parse_args()
    return args



def print_args(args):
    for k, v in vars(args).items():
        print('{:<16} : {}'.format(k, v))