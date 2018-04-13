import requests
import datetime
from glob import glob
from multiprocessing import Pool
import pandas as pd
import pickle
import numpy as np
def download(day):
    url = 'http://beijingair.sinaapp.com/data/beijing/{}/{}/csv'.format('extra', day)  
    r = requests.get(url)
    with open('extra_data/{}_{}.csv'.format('extra', day), 'wb') as f:  
        f.write(r.content)
    url = 'http://beijingair.sinaapp.com/data/beijing/{}/{}/csv'.format('all', day)  
    r = requests.get(url)
    with open('extra_data/{}_{}.csv'.format('all', day), 'wb') as f:  
        f.write(r.content)

def crawl():
    start = datetime.date(2018, 1, 1)
    end = datetime.date(2018, 1, 31)
    numdays = (end - start).days 
    date_list = [(end - datetime.timedelta(days=x)).strftime('%Y%m%d')\
        for x in range(0, numdays)]
    with Pool(processes=8) as p:
        p.map(download, date_list)

def load_pkl(path):
    return pickle.load(open( path, 'rb'))	

def dump_pkl(file, path):
	pickle.dump(file, open( path, 'wb'))	

def local2utc(date):
    year = int(date[0:4])
    month = int(date[4:6])
    day = int(date[6:8])
    hour = int(date[8:10])
    d = datetime.datetime(year, month, day, hour)
    d = d - datetime.timedelta(seconds=8 * 60* 60)
    d = d.strftime('%Y-%m-%d %H:%M:%S')
    return d
    
    
schema = ['PM2.5','PM10','NO2','CO','O3','SO2']


en2ch = load_pkl('meta/en2ch.pkl')
print(en2ch)
ch2en = load_pkl('meta/ch2en.pkl')
print(ch2en)

result = []
if __name__ == "__main__":
    for year in [2013, 2014, 2015, 2016, 2017, 2018]:
        e_list = glob('extra_data/extra_{}*.csv'.format(year))
        e_list.sort()
        a_list = glob('extra_data/all_{}*.csv'.format(year))
        a_list.sort()
        D = {}
        for a, e in zip(a_list, e_list):
            a_df = pd.read_csv(a)
            a_df = a_df.loc[a_df['type'].isin(schema)]
            e_df = pd.read_csv(e)
            e_df = e_df.loc[e_df['type'].isin(schema)]
            df = pd.concat([a_df, e_df])
            df['local_time'] = (df['date'] * 100 + df['hour']).astype(str)
        # df = df.transpose()
            for k, v in ch2en.items():
                col = df[['local_time', 'type', k]]
                if (v not in D):
                    D[v] = {}
                i = 0
                temp = []
                for m in schema:
                    row = col.loc[col['type'] == m]
                    if (i == 0) :
                        date = row['local_time'].values.tolist()
                        if ('local_time' not in D[v]):
                            D[v]['local_time'] = []
                        D[v]['local_time'] += date
                    i += 1
                    value = row[k].values.tolist()
                    if (m not in D[v]):
                        D[v][m] = []
                    D[v][m] += value
                
        schema = ['stationId', 'utc_time'] + schema
        stations = list(D.keys())
        stations.sort()
        result = pd.DataFrame(columns=schema)
        for s in stations:
            M = D[s]
            M['stationId'] = [s + '_aq' for i in range(len(M['local_time']))]
            with Pool(processes=8) as p:
                M['utc_time'] = p.map(local2utc, M['local_time'])
            df = pd.DataFrame(data=M)
            df = df[schema]
            result = result.append(df)
        result.to_csv('extra_data/{}.csv'.format(year), sep=',', index=False)
                    

        
        
        
    


