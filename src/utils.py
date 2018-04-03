# -*- coding: utf-8 -
import numpy as np 
import pandas as pd
import pickle
from random import random, shuffle
import string
schema = ['PM2.5','PM10', 'NO2','CO','O3','SO2']

def load_pkl(path):
	return pickle.load(open( path, 'rb'))	

def dump_pkl(file, path):
	pickle.dump(file, open( path, 'wb'))	

def load_npy(path):
	file = np.load(path)	
	return file 

def dump_npy(file, path):
	np.save(path, file)	

def load_txt(path):
	return np.genfromtxt(path, dtype='str')

def dump_txt(file, path):
	np.savetxt(path, file, delimiter=',')


def load_meta(meta_dir):
    en2ch = load_pkl('{}/en2ch.pkl'.format(meta_dir))
    ch2en = load_pkl('{}/ch2en.pkl'.format(meta_dir))
    station_info = load_pkl('{}/station_info.pkl'.format(meta_dir))
    return en2ch, station_info
    pass

def load_train_data(args):
    df = pd.read_csv(args.train_path)
    en2ch, station_info = load_meta(args.meta_dir)

    areas = {}
    for a, a_v in station_info.items():
        area = {'area': a}
        for s, s_v in a_v.items():
            areas[s] = {'loc' : s_v, 'area': a} 
    df['utc_time'] = pd.to_datetime(df['utc_time'])

    df['year'] = df['utc_time'].dt.year
    df['quarter'] = df['utc_time'].dt.quarter
    df['month'] = df['utc_time'].dt.month
    df['week'] = df['utc_time'].dt.week
    df['weekday'] = df['utc_time'].apply(lambda x: x.weekday())
    df['date'] = df['utc_time'].dt.date
    df['day'] = df['utc_time'].dt.day
    df['hour'] = df['utc_time'].dt.hour

    df['area'] = df['stationId'].apply(lambda x: areas[x]['area'])
    print(len(df) - len(df.dropna()))
    df = df.interpolate(limit=1, limit_direction='both')
    stations = df['stationId'].unique()
    years = df['year'].unique()
    weeks = df['week'].unique()
    hours = df['hour'].unique()
    print(len(df) - len(df.dropna()))
    print( df.isnull().any())
    
    for s in stations:
        s_loc = df['stationId'] == s
        a = areas[s]['area']
        a_loc = df['area'] == a
        for y in years:
            y_loc = df['year'] == y
            for w in weeks:
                w_loc = df['week'] == w
                for h in hours:
                    h_loc = df['hour'] == h
                    loc = s_loc & y_loc & w_loc & h_loc
                    if (len(df.loc[loc].dropna()) == 0):
                        loc = a_loc & y_loc & w_loc & h_loc
                        if(len(df.loc[loc].dropna()) == 0):
                            loc = s_loc & y_loc & h_loc
                            if(len(df.loc[loc].dropna()) == 0):
                                loc = a_loc & y_loc & h_loc
                                if(len(df.loc[loc].dropna()) == 0):
                                    loc =  y_loc & h_loc
                                    
                    mean = df.loc[loc].dropna().mean()
                    df.loc[loc] = df.loc[loc].fillna(mean)
    print( df.isnull().any())
    print(len(df) - len(df.dropna()))
    
    df = df[['stationId', 'utc_time']+schema]
    df.to_csv("data/train_fill.csv", sep=',',index=False)

    return df

def group_mu(df, column):
    mean = df[schema].mean().round(2)
    df = df.groupby(column)
    mu = {}
    for c, d in df:
        d = d.dropna()
        d = d[schema]
        if (len(d) == 0):
            mu[c] = mean
        else :
            mu[c] = d.mean().round(2)
    return mu
        
def split_data(data):
    data = data.values
    schema  = ['stationId', 'utc_time', 'PM2.5','PM10', 'NO2','CO','O3','SO2']
    data_tr = []
    data_te = []
    for row in data:
        if int(row[1].split('-')[2].split(' ')[0]) < 21:
            data_tr.append(row)
        else :
            data_te.append(row)
    return data_tr, data_te

def shuffle_arrs( *arrs):
    r = random()
    result = []
    for arr in arrs:       
        shuffle(arr, lambda : r)   
        result.append(arr)
    return tuple(result)
