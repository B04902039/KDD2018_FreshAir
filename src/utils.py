# -*- coding: utf-8 -
import numpy as np 
import pandas as pd
import pickle
from random import random, shuffle
import string
import math
from torch.utils.data import Dataset, DataLoader
from torch import FloatTensor

schema = ['PM2.5','PM10','O3','NO2','CO','SO2']
d_schema = ['quarter_cos', 'quarter_sin', 'month_cos', 'month_sin', 
    'weekday_cos', 'weekday_sin', 'hour_cos', 'hour_sin']
a_schema = ['other',  'suburban', 'traffic', 'urban']
class AirDataSet(Dataset):
    def __init__(self, data = None, istrain = True):
        self.inputs = data['input']
        self.istrain = istrain
        if(self.istrain) :
            self.outputs = data['output']

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = FloatTensor(self.inputs[idx])
        if(self.istrain) :
            output = FloatTensor(self.outputs[idx])
            sample = {'input': input, 'output': output}
        else :
            sample = {'input': input}
        return sample

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
    areas = load_pkl('{}/areas.pkl'.format(meta_dir))
    return en2ch, ch2en, station_info, areas
    pass

def load_train_data(args):
    df = pd.read_csv(args.train_path)
    en2ch, ch2en, station_info, areas = load_meta(args.meta_dir)
    df = parse_time(df, areas)
    # df = fillna(df, areas)
    df = df.fillna(method='pad')
    df = df.fillna(0)
    
    area = pd.get_dummies(df['area'])
    df = df[['stationId'] + schema + ['date','quarter','month','weekday','hour']]
    df = df.join(area)
    df = tranform_corordinate(df)
    df = df.groupby('stationId')
    data = {}

    return hour2hour_nogroup(df)

def hour2hour_nogroup(df):
    data = {}
    for g1, s in df:
        data[g1] = {}
        s = s.groupby('date')
        fs = []
        hs = []
        inputs = []
        outputs = []
        for g2, d in s:
            f = d[schema+ d_schema + a_schema].values
            fs.append(f)
            h = d[['PM2.5','PM10','O3']].values
            hs.append(h)
        fs = fs[:-2]
        hs = hs[2:]
        for i in range(len(fs) - 1):
            f = np.concatenate((fs[i], fs[i+1]), axis=0)
            inputs.append(f)
            h = hs[i]
            # h = np.concatenate((hs[i], hs[i+1]), axis=0)
            outputs.append(h)
        data[g1]['input'] = np.asarray(inputs)        
        data[g1]['output'] = np.asarray(outputs)
        
    return data

def hour2hour_nogroup(df):
    data = {}
    for g1, d in df:
        data[g1] = {}
        fs = []
        hs = []
        inputs = []
        outputs = []
        fs = d[schema+ d_schema + a_schema].values
        hs = d[['PM2.5','PM10','O3']].values
        fs = fs[:-48]
        hs = hs[48:]
        for i in range(len(fs) - 47):
            f =fs[i:i+48]
            inputs.append(f)
            h =hs[i:i+24]
            outputs.append(h)
        data[g1]['input'] = np.asarray(inputs)
        data[g1]['output'] = np.asarray(outputs)
    return data

def date2date(df):
    data = {}
    for g1, s in df:
        data[g1] = {}
        s = s.groupby('date')
        fs = []
        hs = []
        for g2, d in s:
            f = np.mean(d[schema+ d_schema+ a_schema].values, 0)
            fs.append(f)
            h = np.mean(d[['PM2.5','PM10','O3']].values, 0)
            hs.append(h)
        fs = np.asarray(fs)
        hs = np.asarray(hs[10:])
        inputs = []
        outputs = []
        for i in range(len(hs) - 1):
            inputs.append(fs[i:i+10])
            outputs.append(hs[i:i+2])
        data[g1]['input'] = np.asarray(inputs)
        data[g1]['output'] = np.asarray(outputs)
    return data

def hour2date(df):
    data = {}
    for g1, s in df:
        data[g1] = {}
        s = s.groupby('date')
        fs = []
        hs = []
        for g2, d in s:
            f = d[schema+ d_schema+ a_schema].values
            fs.append(f)
            h = np.mean(d[['PM2.5','PM10','O3']].values, 0)
            hs.append(h)
        fs = np.asarray(fs[:-2])
        hs = np.asarray(hs[2:])
        inputs = []
        outputs = []
        for i in range(len(fs) - 1):
            f = np.concatenate((fs[i], fs[i+1]), axis=0)
            inputs.append(f)
            h = hs[i:i+2]
            outputs.append(h)
        data[g1]['input'] = inputs
        data[g1]['output'] = outputs
    return data

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
    df['area'] = df['stationId'].apply(lambda x: areas[x]['area'])
    return df
def tranform_corordinate(df):
    df['quarter_cos'], df['quarter_sin'] = to_polor(df['quarter'], 4)
    df['month_cos'], df['month_sin'] = to_polor(df['month'], 31)
    df['weekday_cos'], df['weekday_sin'] = to_polor(df['weekday'], 7)
    df['hour_cos'], df['hour_sin'] = to_polor(df['hour'], 24)
    return df

def to_polor(col, period):
    col1 = col.apply(lambda x: math.cos(2*math.pi *x /period))
    col2 = col.apply(lambda x: math.sin(2*math.pi *x /period))
    return col1, col2
    
def fillna(df, areas):
    df = df.interpolate(limit=1, limit_direction='both')
    stations = df['stationId'].unique()
    stations = df['area'].unique()
    years = df['year'].unique()
    weeks = df['week'].unique()
    dates = df['date'].unique()
    hours = df['hour'].unique()
    print (df.isnull().values.any())

    for a in areas:
        a_loc = df['area'] == a
        for h in hours:
            h_loc = df['hour'] == h
            for d in dates:
                d_loc = df['date'] == d
                fill = a_loc & d_loc & h_loc
                if (df.loc[fill].isnull().values.any()):
                    loc = fill
                    if(len(df.loc[loc].dropna()) != 0):
                        mean = df.loc[loc].dropna().mean().round(2)
                        df.loc[fill] = df.loc[fill].fillna(mean)

    for s in stations:
        s_loc = df['stationId'] == s
        a = areas[s]['area']
        a_loc = df['area'] == a
        for h in hours:
            h_loc = df['hour'] == h
            for y in years:
                y_loc = df['year'] == y
                for w in weeks:
                    w_loc = df['week'] == w
                    fill = s_loc & y_loc & w_loc & h_loc
                    if (df.loc[fill].isnull().values.any()):
                        loc = fill
                        if (len(df.loc[loc].dropna()) == 0):
                            loc = a_loc & y_loc & w_loc & h_loc
                        if (len(df.loc[loc].dropna()) == 0):
                            loc = s_loc & y_loc & h_loc
                        if (len(df.loc[loc].dropna()) == 0):
                            loc = a_loc & y_loc & h_loc
                        if (len(df.loc[loc].dropna()) == 0):
                            loc = y_loc & h_loc
                        if (len(df.loc[loc].dropna()) != 0):
                            print(a, s, y, w, h)
                            mean = df.loc[loc].dropna().mean().round(2)
                            df.loc[fill] = df.loc[fill].fillna(mean)
    print (df.isnull().values.any())
    
    return df
   
def split_data(data, r):
    valid = int(len(data['input']) * (1 - r))
    data_tr = {}
    data_val = {}
    data_tr['input'], data_val['input'] = data['input'][:valid], data['input'][valid:]
    data_tr['output'], data_val['output'] = data['output'][:valid], data['output'][valid:]
    
    return data_tr, data_val

def shuffle_arrs( *arrs):
    r = random()
    result = []
    for arr in arrs:       
        shuffle(arr, lambda : r)   
        result.append(arr)
    return tuple(result)
