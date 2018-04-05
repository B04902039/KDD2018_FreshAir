# -*- coding: utf-8 -
import os
from arguments import *
from model import *
from utils import * 

def train(args):
    data = load_train_data(args)
    for g, d in data.items():
        input_dim = d['input'][0].shape[1]
        output_dim = d['output'][0].shape[1]
        model = AirNet(input_dim, output_dim, args)
        train_data, valid_data = split_data(d, args.valid_ratio)
        train_data = AirDataSet(train_data)
        valid_data = AirDataSet(valid_data)
        train_loader = DataLoader(dataset = train_data, 
                            batch_size = args.batch_size, 
                            shuffle = True, 
                            num_workers = 2,
                            pin_memory = args.use_cuda)
        valid_loader = DataLoader(  dataset = valid_data, 
                            batch_size = args.batch_size,
                            shuffle = True, 
                            num_workers = 2,
                            pin_memory = args.use_cuda)
        if (args.load):   
            model.load(args.model_path)
        model.train(train_loader, valid_loader)
        
def test(args):
    pass
    
if __name__ == '__main__':
    if not os.path.exists('meta/'):
        os.makedirs('meta/')
    if not os.path.exists('model/'):
        os.makedirs('model/')
    if not os.path.exists('data/'):
        os.makedirs('data/')
    args = parse()
    print_args(args)
    if args.train:
        train(args)
    if args.test:
        test(args)
        