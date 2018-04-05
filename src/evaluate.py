# -*- coding: utf-8 -
import numpy as np 

def smape(y_true, y_pred):
    diff = np.abs(y_pred - y_true)
    factor = (np.abs(y_pred) + np.abs(y_true))
    y = diff / factor
    y[np.isnan(y)] = 0
    return 200 * np.mean(y)

def mape(y_true, y_pred):
    diff = np.abs(y_pred - y_true)
    factor = (np.abs(y_true))
    y = diff / factor
    y[np.isnan(y)] = 0
    return 100 * np.mean(y)

def mse(y_true, y_pred):
    diff = np.square(y_pred - y_true)
    return np.mean(diff)

def mae(y_true, y_pred):
    diff = np.abs(y_pred - y_true)
    return np.mean(diff)