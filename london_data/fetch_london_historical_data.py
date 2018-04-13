#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 17:43:12 2018

@author: luo
"""

import os
import subprocess

stations = ['BX9', 'BX1', 'BL0', 'CD9', 'CD1', 'CT2', 'CT3', 'CR8', 'GN0',
            'GR4', 'GN3', 'GR9', 'GB0', 'HR1', 'HV1', 'LH0', 'KC1', 'KF1',
            'LW2', 'RB7', 'TD5', 'ST5', 'TH4', 'MY7']

prefix = '\'www.londonair.org.uk/london/asp/downloadsite.asp?site='
profix1 = '&species1=COm&species2=NOm&species3=NO2m&species4=NOXm&species5=O3m&species6=PM10m&'
profix2 = '&species1=PM25m&species2=SO2m&species3=&species4=&species5=&species6=&'
# time_span format: 'start=day-month-year&end=day-month-year&res=6&period=hourly&units=ugm3\'
time_span = 'start=1-jan-2011&end=1-jan-2013&res=23&period=hourly&units=ugm3\''


for i, s in enumerate(stations):
    print(i, s)
    
    link1 = '{}{}{}{}'.format(prefix, s, profix1, time_span)
    subprocess.run('curl {} > {}_1.csv'.format(link1, s), shell=True)
    link2 = '{}{}{}{}'.format(prefix, s, profix2, time_span)
    subprocess.run('curl {} > {}_2.csv'.format(link2, s), shell=True)
    
    P = subprocess.run(['tail', '{}_2.csv'.format(s)], stdout=subprocess.PIPE)
    print(P)