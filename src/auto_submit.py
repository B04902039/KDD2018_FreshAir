#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 23:41:32 2018

@author: luo
"""

import requests
import os
from datetime import datetime
import subprocess
import pandas as pd
import numpy as np

from update_latest_data import downloadLatestData

def api_submit(filepath):
    files={'files': open(filepath,'rb')}
    print(files)
    
    time_stamp = str(datetime.now())
    data = {
        "user_id": "LoChiChen",   #user_id is your username which can be found on the top-right corner on our website when you logged in.
        "team_token": "f62b378aef32f97c0eb7eda878af2fe6bf7e8dc16ab76f2aedfb55522ba15967", #your team_token.
        "description": time_stamp,  #no more than 40 chars.
        "filename": filepath, #your filename
    }
    print(data)
    
    url = 'https://biendata.com/competition/kdd_2018_submit/'
    response = requests.post(url, files=files, data=data)
    print(response.text)
    
    with open('submission_log.txt', 'a') as f:
        f.write('{}:\t{}\n{}\n\n'.format(time_stamp, filepath, response.text))

    
def make_baseline_submission_file(sample, bj, ld):
    stations = list(sample.test_id.apply(lambda x: x.split(sep='#')[0]).unique())
    sample = np.array(sample)
    stations = [x[:7] for x in stations]
    ld_stations = stations[-13:]
    bj_stations = stations[:-13]
    for i, s in enumerate(bj_stations):
        tmp = bj[bj.apply(lambda x: s in x['stationId'], axis=1)]
        pm25_means = tmp['PM2.5'].mean()
        pm10_means = tmp['PM10'].mean()
        O3_means = tmp['O3'].mean()
        date = tmp['day'].values[-1]
        today = tmp[tmp['day']==date]
        today = np.array(today)[:, (2,3,5)]
        tmp = [x for x in sample if s in x[0]]
        for i, data in enumerate(today):
            tmp[i][1:] = data
            tmp[i+24][1:] = data
        for i, data in enumerate(tmp):
            if data[1] == 0:
                tmp[i][1:] = [pm25_means, pm10_means, O3_means]
            
    for i, s in enumerate(ld_stations):
        tmp = ld[ld.apply(lambda x: s in x['stationId'], axis=1)]
        pm25_means = tmp['PM2.5'].mean()
        pm10_means = tmp['PM10'].mean()
        O3_means = tmp['O3'].mean()
        date = tmp['day'].values[-1]
        today = tmp[tmp['day']==date]
        today = np.array(today)[:, (2,3,5)]
        tmp = [x for x in sample if s in x[0]]
        for i, data in enumerate(today):
            tmp[i][1:] = data
            tmp[i+24][1:] = data
        for i, data in enumerate(tmp):
            if data[1] == 0:
                tmp[i][1:] = [pm25_means, pm10_means, O3_means]
    
    sample = pd.DataFrame(sample)
    sample.columns = ['test_id', 'PM2.5', 'PM10', 'O3']
    sample.to_csv('../queue/default.csv', index=False)
    return sample

    
if __name__ == '__main__':
    submission_dir = '../queue/'
    submitted_dir = '../submitted/'
    file_names = os.listdir(submission_dir)
    file_names = [f for f in file_names if '.csv' in f]
    
    submission_cnt = 0
    for i, name in enumerate(file_names):
        filepath = '{}{}'.format(submission_dir, name)
        api_submit(filepath)
        subprocess.run('mv {}{} {}{}'.format(submission_dir, name, submitted_dir, name))
        submission_cnt += 1
    
    if submission_cnt < 3:
        #[bj_filepath, ld_filepath] = downloadLatestData()
        ld_fillna_path = '../data/london_latest_fillna.csv'
        bj_fillna_path = '../data/beijing_latest_fillna.csv'
        #subprocess.run('python dataUtil.py {} {} bj'.format(bj_filepath, bj_fillna_path), shell=True)
        #subprocess.run('python dataUtil.py {} {} ld'.format(ld_filepath, ld_fillna_path), shell=True)
        ld_df = pd.read_csv(ld_fillna_path)[-120*13:]
        bj_df = pd.read_csv(bj_fillna_path)[-120*35:]
        sample = pd.read_csv('../data/sample_submission.csv')
        ret = make_baseline_submission_file(sample, bj_df, ld_df)
        api_submit('../queue/default.csv')
        subprocess.run('mv ../queue/default.csv ../submitted/default.csv', shell=True)
        submission_cnt += 1
