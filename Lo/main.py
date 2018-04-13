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
    # drop stationId and utc_time
    try:
        ret = raw.drop(['stationId', 'utc_time'], axis=1)
    except ValueError:
        print('column \'stationId\', \'utc_time\' not exist')
    
    # filling missing data
    ret = ret.fillna(method='bfill')
    ret = ret.fillna(method='ffill')
    
    # prepare data of past hours:
    past = list()
    for i in range(hr, 0, -1):
        past.append(ret.shift(i))
    agg = pd.concat(past, axis=1)
    agg.dropna(inplace=True)
    
    return agg.values, ret.values[hr:, :3]

def SMAPE(y_true, y_pred):
    diff = 2 * K.abs(y_true - y_pred) / K.clip((K.abs(y_true) + K.abs(y_true)), K.epsilon(), None)
    return 100. * K.mean(diff, axis=-1)

if __name__ == "__main__":
    data_path = '../FreshAir_data/stations/';    raw = readDatafromDir(data_path)
    #data_path = '../FreshAir_data/sortedByStation.csv'; raw = readDatafromFile(data_path)
    
    #test on only one station (aotizhongxin):
    #raw = raw.drop('Unnamed: 0',axis=1)
    #raw = raw[raw.stationId == 'aotizhongxin_aq']
    
    sample = 48
    X = []
    Y = []
    for i in range(35):
        raw[i] = raw[i].drop('Unnamed: 0',axis=1)
        x, y = preprocessing(raw[i], sample)
        X.append(x)
        Y.append(y)
    X = np.concatenate(X, axis=1)
    Y = np.concatenate(Y, axis=1)
    #X, Y = preprocessing(raw, hr = sample)
    #print(X[:10])
    #print(Y[:10])
    
    # split training and testing set
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9, random_state=25)
    x_train = x_train.reshape(x_train.shape[0], sample, 6*35)
    x_test = x_test.reshape(x_test.shape[0], sample, 6*35)
    del X
    del Y
    
    # model
    model = Sequential()
    model.add(LSTM(256, input_shape=(x_train.shape[1], x_train.shape[2]), 
                   dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(128))
    model.add(Dropout(0.2))
    model.add(Dense(105))
    model.compile(loss=SMAPE, optimizer='adam')
    
    # train
    filepath="../FreshAir_data/multistation_lstm_dense_48hr_weight.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                                 save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    callbacks_list = [checkpoint, early]
    batch_size = 128
    epochs = 10    
    model.summary()
    #history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, 
                  #validation_data=(x_test, y_test), callbacks=callbacks_list, verbose=1)
    
    # ploting
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()