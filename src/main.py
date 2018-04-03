# -*- coding: utf-8 -
import os
import numpy as np 
from arguments import *
from model import *
from utils import * 

def train(args):
    data = load_train_data(args)

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
        