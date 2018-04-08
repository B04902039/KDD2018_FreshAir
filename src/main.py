# -*- coding: utf-8 -
import os
from arguments import *
from model import *
from utils import * 

def train(args):
    data = load_train_data(args)
    for g, d in data.items():
    # if True :
        # d = data
        model = AirNet(d['input'][0].shape, d['output'][0].shape, args)
        # model = AirNet(d['input'][0].shape, d['output'][0].shape, args)
        
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
        tr_hist, val_hist = model.train(train_loader, valid_loader)
        np.save('{}/{}_{}_tr.npy'.format(args.exp_dir, args.decoder_type, g), tr_hist)
        np.save('{}/{}_{}_val.npy'.format(args.exp_dir, args.decoder_type, g), val_hist)
        
def test(args):
    pass
    
if __name__ == '__main__':
    if not os.path.exists('meta/'):
        os.makedirs('meta/')
    if not os.path.exists('model/'):
        os.makedirs('model/')
    if not os.path.exists('data/'):
        os.makedirs('data/')
    if not os.path.exists('exp/'):
        os.makedirs('exp/')
    args = parse()
    print_args(args)
    if args.train:
        train(args)
    if args.test:
        test(args)
        