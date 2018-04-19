#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 10:05:58 2018

@author: luo
"""

import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU
from keras.callbacks import Callback
from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.preprocessing import text, sequence
from keras.layers import GlobalAveragePooling1D, concatenate, SpatialDropout1D
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Model, Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K

def readDatafromDir(path):
    file_pathes = os.listdir(path)
    ret = []
    for i in file_pathes:
        ret.append(pd.read_csv(path+i))    
    return ret

def readDatafromFile(path):
    return pd.read_csv(path)

def preprocessing(raw, hr=9):
    area = ['urban']#, 'suburban', 'traffic', 'other']
    groups = []
    y = []
    for a in area:
        ret = raw[raw['area']==a]
        ret = ret.drop(['stationId', 'utc_time', 'date'], axis=1)
        past = list()
        for i in range(hr, 0, -1):
            past.append(ret.shift(i))
            agg = pd.concat(past, axis=1)
            agg.dropna(inplace=True)
        groups.append(agg.values)
        y.append(ret.values[hr:, :3])
        
    return groups, y

def SMAPE(y_true, y_pred):
    diff = 2 * K.abs(y_true - y_pred) / K.clip((y_true + y_pred), K.epsilon(), None)
    return 100. * K.mean(diff, axis=-1)

if __name__ == "__main__":
    data_path = '../data/train_fillna.csv'
    raw = readDatafromFile(data_path)
    #data_path = '../../FreshAir_data/sortedByStation.csv'; raw = readDatafromFile(data_path)
    
    #test on only one station (aotizhongxin):
    #raw = raw.drop('Unnamed: 0',axis=1)
    #raw = raw[raw.stationId == 'aotizhongxin_aq']
    
    sample = 48

    X, Y = preprocessing(raw, sample)
    #X, Y = preprocessing(raw, hr = sample)
    #print(X[:10])
    #print(Y[:10])
    
    '''# split training and testing set
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9, random_state=25)
    x_train = x_train.reshape(x_train.shape[0], sample, 6*35)
    x_test = x_test.reshape(x_test.shape[0], sample, 6*35)
    del X
    del Y
    
    # model
    model = Sequential()
    model.add(Dense(256, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Dense(105))
    model.compile(loss='mae', optimizer='adam')
    
    # train
    filepath="../FreshAir_data/dense_48hr_weight.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                                 save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    callbacks_list = [checkpoint, early]
    batch_size = 128
    epochs = 10    
    model.summary()
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, 
                  validation_data=(x_test, y_test), callbacks=callbacks_list, verbose=1)
    
    # ploting
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    '''