# -*- coding: utf-8 -
import numpy as np 
import pandas as pd
import sys
from multiprocessing import Pool



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

def fillna(df, meta_dir):
    areas = load_pkl('{}/areas.pkl'.format(meta_dir))
    df = parse_time(df, areas)
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
def build_feat(x):
    t, t_df = x
    feat = np.zeros([lon, lat, len(schema)])
    t_df = t_df.groupby('longitude')
    for l, l_df in t_df:
        l_df = l_df.groupby('latitude')
        for w, w_df in l_df:
            feat[l, w] = w_df[schema].values[0]
    print('B', t, feat.shape)
    return feat

if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    df = pd.read_csv(input_path)
    schema = list(df.columns.values)[4:]
    lon_bound = (df['longitude'].min(0), df['longitude'].max(0))
    lat_bound = (df['latitude'].min(0), df['latitude'].max(0))
    df['latitude'] = df['latitude'].apply(lambda x: int((x - lat_bound[0]) / 0.1))
    df['longitude'] = df['longitude'].apply(lambda x: int((x - lon_bound[0]) / 0.1))
    lon = int((lon_bound[1] - lon_bound[0]) / 0.1) +1
    lat = int((lat_bound[1] - lat_bound[0]) / 0.1) +1
    df = df.groupby('utc_time')

    with Pool(processes=16) as p:
        data = list(p.map(build_feat, df))
    data = np.asarray(data)
    print(data.shape)
    np.save(output_path, data)