#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 11:08:17 2018

@author: luo
"""

import numpy as np 
import pandas as pd
import sys

beijing2area = {
    'dongsi_aq': 'urban',
    'tiantan_aq': 'urban',
    'guanyuan_aq':'urban',
    'wanshouxigong_aq':'urban',
    'aotizhongxin_aq':'urban',
    'nongzhanguan_aq':'urban',
    'wanliu_aq':'urban',
    'beibuxinqu_aq':'urban',
    'zhiwuyuan_aq':'urban',
    'fengtaihuayuan_aq':'urban',
    'yungang_aq':'urban',
    'gucheng_aq':'urban',
    'fangshan_aq':'suburban',
    'daxing_aq':'suburban',
    'yizhuang_aq':'suburban',
    'tongzhou_aq':'suburban',
    'shunyi_aq':'suburban',
    'pingchang_aq':'suburban',
    'mentougou_aq':'suburban',
    'pinggu_aq':'suburban',
    'huairou_aq':'suburban',
    'miyun_aq':'suburban',
    'yanqin_aq':'suburban',
    'dingling_aq':'other',
    'badaling_aq':'other',
    'miyunshuiku_aq':'other',
    'donggaocun_aq':'other',
    'yongledian_aq':'other',
    'yufa_aq':'other',
    'liulihe_aq':'other',
    'qianmen_aq':'traffic',
    'yongdingmennei_aq': 'traffic',
    'xizhimenbei_aq': 'traffic',
    'nansanhuan_aq': 'traffic',
    'dongsihuan_aq': 'traffic'
}

london2area = {
    'BX9':'Suburban',
    'BX1':'Suburban',
    'BL0':'Urban',
    'CD9':'Roadside',
    'CD1':'Kerbside',
    'CT2':'Kerbside',
    'CT3':'Urban',
    'CR8':'Urban',
    'GN0':'Roadside',
    'GR4':'Suburban',
    'GN3':'Roadside',
    'GR9':'Roadside',
    'GB0':'Roadside',
    'HR1':'Urban',
    'HV1':'Roadside',
    'LH0':'Urban',
    'KC1':'Urban',
    'KF1':'Urban',
    'LW2':'Roadside',
    'RB7':'Urban',
    'TD5':'Suburban',
    'ST5':'Industrial',
    'TH4':'Roadside',
    'MY7':'Kerbside'
}

def parse_time(df, areas):
    df['utc_time'] = pd.to_datetime(df['utc_time'])
    df['year'] = df['utc_time'].dt.year
    df['quarter'] = df['utc_time'].dt.quarter
    df['month'] = df['utc_time'].dt.month
    df['week'] = df['utc_time'].dt.week
    df['weekday'] = df['utc_time'].apply(lambda x: x.weekday())
    df['date'] = df['utc_time'].dt.date
    df['day'] = df['utc_time'].dt.day
    df['hour'] = df['utc_time'].dt.hour
    df['area'] = df['stationId'].apply(lambda x: areas[x])
    return df   