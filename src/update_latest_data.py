#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 17:03:28 2018

@author: luo
"""

import subprocess
import pandas as pd

def downloadLatestData():
    csv_cols = ['stationId', 'utc_time', 'PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
    ld_url = 'https://biendata.com/competition/airquality/ld/2018-01-01-0/2019-01-01-23/2k0d1d8'
    bj_url = 'https://biendata.com/competition/airquality/bj/2018-01-01-0/2019-01-01-23/2k0d1d8'
    ld_filepath = '../data/london_latest'
    bj_filepath = '../data/beijing_latest'
    subprocess.run('wget {} -O {}.csv'.format(ld_url, ld_filepath), shell=True)
    subprocess.run('wget {} -O {}.csv'.format(bj_url, bj_filepath), shell=True)
    
    df = pd.read_csv(ld_filepath+'.csv')
    df = df.drop('id', axis=1)
    df.columns = csv_cols
    date = df['utc_time'][len(df)-1]
    subprocess.run('rm {}*'.format(ld_filepath), shell=True)
    ld_filepath = '{}{}.csv'.format(ld_filepath,date.replace(' ', '-'))
    df.to_csv(ld_filepath, index=False)
    
    df = pd.read_csv(bj_filepath+'.csv')
    df = df.drop('id', axis=1)
    df.columns = csv_cols
    date = df['utc_time'][len(df)-1]
    subprocess.run('rm {}*'.format(bj_filepath), shell=True)
    bj_filepath = '{}{}.csv'.format(bj_filepath,date.replace(' ', '-'))
    df.to_csv(bj_filepath, index=False)
    
    return [bj_filepath, ld_filepath]
    
if __name__ == '__main__':
    print(downloadLatestData())