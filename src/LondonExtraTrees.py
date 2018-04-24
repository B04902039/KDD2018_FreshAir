import numpy as np
import pandas as pd
from utils import *
from dataset import *
from evaluate import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.externals import joblib
from constant import *
import sys

df_train = pd.read_csv(sys.argv[1])
df_val = pd.read_csv(sys.argv[2])
df_test = pd.read_csv(sys.argv[3])
df_train = df_train.sort_values(['year', 'month', 'day', 'hour'])
df_val = df_val.sort_values(['year', 'month', 'day', 'hour'])
df_test = df_test.sort_values(['year', 'month', 'day', 'hour'])
#stationId = [s for s in df_train.stationId.unique()]
stationId = citymeta['london']['schema']['station']
drop = ['utc_time' , 'year', 'quarter', 'month', 'week', 'weekday', 'date', 'day', 'hour', 'area', 'NO', 'NOX', 'CO', 'O3', 'SO2']
drop_ = ['utc_time' , 'year', 'quarter', 'month', 'week', 'weekday', 'date', 'day', 'hour', 'area', 'CO', 'O3', 'SO2']
df_train = tranform_corordinate(df_train)
df_val = tranform_corordinate(df_val)
df_test = tranform_corordinate(df_test)
#df_train = df_train.fillna(df_train.mean())
#df_val = df_val.fillna(df_val.mean())
#df_test = df_test.fillna(df_test.mean())
#df_test = tranform_corordinate(df_test)
df_train = df_train.drop(drop, axis = 1)
df_val = df_val.drop(drop, axis = 1)
df_test = df_test.drop(drop_, axis = 1)
df_train = pd.concat([df_train, df_val])
df_train = df_train.fillna(df_train.mean())
df_test = df_test.fillna(df_test.mean())
df_train = df_train.groupby(['stationId'])
#df_val = df_val.groupby(['stationId'])
df_test = df_test.groupby(['stationId'])

for s in stationId:
  print(s)
  train = df_train.get_group(s).drop(['stationId'], axis = 1)
  IDX = list(train)
  #val = df_val.get_group(s).drop(['stationId'], axis = 1)
  test = df_test.get_group(s).drop(['stationId'], axis = 1)
  train = train.values 
  #val = val.values
  test = test.values
  X = np.zeros((train.shape[0] - 23, 24 * train.shape[1]))
  #X_ = np.zeros((val.shape[0] - 23, 24 * val.shape[1]))
  X_ = np.zeros((test.shape[0] - 23, 24 * test.shape[1]))
  for i in range(X.shape[0]):
    for j in range(24):
      X[i, train.shape[1] * j: train.shape[1] * (j + 1)] = train[i + j, :] 
  for i in range(X_.shape[0]):
    for j in range(24):
      #X_[i, val.shape[1] * j: val.shape[1] * (j + 1)] = val[i + j, :]
      X_[i, test.shape[1] * j: test.shape[1] * (j + 1)] = test[i + j, :]

  X_train = X[:-48]
  y_train = np.zeros((X_train.shape[0], 48 * 2))
  for i in range(y_train.shape[0]):
    for j in range(24):
      y_train[i, 2 * j] = X[i + 24, train.shape[1] * j]
      y_train[i, 2 * j + 1] = X[i + 24, train.shape[1] * j + 1]
      #y_train[i, 3 * j + 2] = X[i + 24, train.shape[1] * j + 4]
      y_train[i, 2 * j + 48] = X[i + 48, train.shape[1] * j]
      y_train[i, 2 * j + 48 + 1] = X[i + 48, train.shape[1] * j + 1]
      #y_train[i, 3 * j + 72 + 2] = X[i + 48, train.shape[1] * j + 4]
  """
  X_val = X_[:-48]
  y_val = np.zeros((X_val.shape[0], 48 * 2))
  for i in range(y_val.shape[0]):
    for j in range(24):
      y_val[i, 2 * j] = X_[i + 24, val.shape[1] * j]
      y_val[i, 2 * j + 1] = X_[i + 24, val.shape[1] * j + 1]
      #y_val[i, 3 * j + 2] = X_[i + 24, val.shape[1] * j + 4]
      y_val[i, 2 * j + 48] = X_[i + 48, val.shape[1] * j]
      y_val[i, 2 * j + 48 + 1] = X_[i + 48, val.shape[1] * j + 1]
      #y_val[i, 3 * j + 72 + 2] = X_[i + 48, val.shape[1] * j + 4]
      #y_val[i, j + 24] = X_[i + 48, val.shape[1] * j]
  """
  X_test = X_[:-48]
  y_test = np.zeros((X_test.shape[0], 48 * 2))
  for i in range(y_test.shape[0]):
    for j in range(24):
      y_test[i, 2 * j] = X_[i + 24, test.shape[1] * j]
      y_test[i, 2 * j + 1] = X_[i + 24, test.shape[1] * j + 1]
      #y_test[i, 3 * j + 2] = X_[i + 24, test.shape[1] * j + 4]
      y_test[i, 2 * j + 48] = X_[i + 48, test.shape[1] * j]
      y_test[i, 2 * j + 48 + 1] = X_[i + 48, test.shape[1] * j + 1]
      #y_test[i, 3 * j + 72 + 2] = X_[i + 48, test.shape[1] * j + 4]


  #parameters = {'n_estimators':[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]}
  #parameters = {'n_estimators':[50, 70, 100, 120, 150]}
  #parameters = {'n_estimators':[2]}
  #regressor = GridSearchCV(ExtraTreesRegressor(random_state=0), parameters)
  #print(regressor.best_params_)
  #regressor = ExtraTreesRegressor(n_estimators = 30)
  regressor = ExtraTreesRegressor(n_estimators = 150)
  regressor.fit(X_train, y_train)
  model_name = '../model/ExtraTreesRegressor_' + s +'.pkl'
  #regressor = joblib.load(model_name)
  #importances = regressor.feature_importances_
  #indices = np.argsort(importances)[::-1]
  #f_num = X_train.shape[1] // 24
  #for f in range(X_train.shape[1]):
  #  print("(%f) %s %d" % (importances[indices[f]] / importances[indices[0]], IDX[indices[f] % f_num], 24 - indices[f] // f_num ))
  #predictions = regressor.predict(X_val)
  predictions = regressor.predict(X_test)
  #print(smape(y_val, predictions))
  print(smape(y_test, predictions))
  print(smape(y_test[:, :48], predictions[:, :48]))
  print(smape(y_test[:, 48:], predictions[:, 48:]))
  #model_name = '../model/ExtraTreesRegressor_' + s +'.pkl'
  joblib.dump(regressor, model_name)
  #print(regressor.best_params_)
