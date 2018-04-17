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

from update_latest_data import downloadLatestData

def api_submit(filepath):
    files={'files': open(filepath,'rb')}
    
    time_stamp = str(datetime.now)
    data = {
        "user_id": "LoChiChen",   #user_id is your username which can be found on the top-right corner on our website when you logged in.
        "team_token": "f62b378aef32f97c0eb7eda878af2fe6bf7e8dc16ab76f2aedfb55522ba15967", #your team_token.
        "description": time_stamp,  #no more than 40 chars.
        "filename": filepath, #your filename
    }
    
    url = 'https://biendata.com/competition/kdd_2018_submit/'
    response = requests.post(url, files=files, data=data)
    
    with open('submission_log.txt', 'a') as f:
        f.write('{}:\t{}\n{]\n\n'.format(time_stamp, filepath, response))        

    
if __name__ == '__main__':
    submission_dir = '../queue/'
    submitted_dir = '../submitted/'
    file_names = os.listdir(submission_dir)
    file_names = [f for f in file_names if '.csv' in f]
    
    submission_cnt = 0
    for i, name in enumerate(file_names):
        filepath = '{]{}'.format(submission_dir, name)
        api_submit(filepath)
        submission_cnt += 1
    
    if submission_cnt < 3:
        [bj_filepath, ld_filepath] = downloadLatestData()
        ld_fillna_path = '../data/london_latest_fillna.csv'
        bj_fillna_path = '../data/beijing_latest_fillna.csv'
        subprocess.run('python dataUtil.py {} {} bj'.format(bj_filepath, bj_fillna_path), shell=True)
        subprocess.run('python dataUtil.py {} {} ld'.format(ld_filepath, ld_fillna_path), shell=True)