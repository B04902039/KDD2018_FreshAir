#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 23:49:58 2018

@author: luo
"""

old = '01/02/2017 10:00'
new = '2017-02-01 10:00:00'

def trans_date(old):
    tmp = old.split(sep=' ')
    date = tmp[0]
    hour = tmp[1]+':00'
    [day, month, year] = date.split(sep='/')
    ret = '{}-{}-{} {}'.format(year, month, day, hour)
    return ret

print(trans_date(old))
