# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 12:18:07 2018

@author: 羅際禎
"""

import pandas as pd
import os


stations = ['BX9', 'BX1', 'BL0', 'CD9', 'CD1', 'CT2', 'CT3', 'CR8', 'GN0',
            'GR4', 'GN3', 'GR9', 'GB0', 'HR1', 'HV1', 'LH0', 'KC1', 'KF1',
            'LW2', 'RB7', 'TD5', 'ST5', 'TH4', 'MY7']

filenames = os.listdir('.')
filenames = [x for x in filenames if x[-3:] == 'csv']

for i, s in enumerate(stations):
    species_order = ['stationId', 'utc_time', 'PM2.5', 'PM10', 'NO2', 'CO',
                     'O3', 'SO2', 'NO', 'NOX']
    print(i,s)
    df = pd.read_csv(s+'.csv')
    df = df.drop(['Unnamed: 0', 'Units', 'Provisional or Ratified'], axis=1)
    df = df.pivot(index='ReadingDateTime', columns='Species', values='Value')
    df = df.reset_index()
    df = df.rename(columns={'ReadingDateTime':'utc_time'})
    df['stationId'] = s
    df = df[species_order]
    df.to_csv('{}m.csv'.format(s), index=False)

df = pd.read_csv('BX9m.csv')
for s in stations[1:]:
    tmp = pd.read_csv(s+'m.csv')
    df = pd.concat([df, tmp])
df.to_csv('london_17_18_aq.csv', index=False)
 
# merge station_1 _2 into one file
for i, s in enumerate(stations):
    tmp1 = '{}_1.csv'.format(s)
    tmp2 = '{}_2.csv'.format(s)
    tmp1 = pd.read_csv(tmp1)
    tmp2 = pd.read_csv(tmp2)
    tmp = pd.concat([tmp1,tmp2], axis=0)
    tmp.to_csv('{}.csv'.format(s))