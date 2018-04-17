# -*- coding: utf-8 -
import os
from arguments import *
from model import *
from utils import * 

def train(args):
    train, test = load_train_data(args)
    train_hour = hour2hour_nogroup(train)
    train_date = hour2hour(train)
    test_date = hour2hour(test)

    for g, d in train_hour.items():
        # for knowing the progress
        print(g,d)
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
        valid_loader = DataLoader(dataset = valid_data, 
                            batch_size = args.batch_size,
                            shuffle = True, 
                            num_workers = 2,
                            pin_memory = args.use_cuda)
        print('==== train ====')
        if (args.load):   
            model.load(args.model_path)
        tr_hist, val_hist = model.train(train_loader, valid_loader)

        print('==== tune ====')
        d = train_date[g]
        train_data, valid_data = split_data(d, args.valid_ratio)
        test_data = test_date[g]
        train_data = AirDataSet(train_data)
        valid_data = AirDataSet(valid_data)
        test_data = AirDataSet(test_data)
        train_loader = DataLoader(dataset = train_data, 
                            batch_size = args.batch_size, 
                            shuffle = True, 
                            num_workers = 2,
                            pin_memory = args.use_cuda)
        valid_loader = DataLoader(dataset = valid_data, 
                    batch_size = args.batch_size,
                    shuffle = True, 
                    num_workers = 2,
                    pin_memory = args.use_cuda)
        test_loader = DataLoader(dataset = test_data, 
                    batch_size = args.batch_size,
                    shuffle = False, 
                    num_workers = 2,
                    pin_memory = args.use_cuda)
        model.train(train_loader, valid_loader)
        print('==== test ====')
        model.eval(test_loader)
        np.save('{}/{}_{}_tr_48.npy'.format(args.exp_dir, args.decoder_type, g), tr_hist)
        np.save('{}/{}_{}_val_48.npy'.format(args.exp_dir, args.decoder_type, g), val_hist)
        
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
        
