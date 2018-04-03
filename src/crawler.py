import requests
import datetime
from glob import glob
from multiprocessing import Pool
import pandas as pd
import pickle
import numpy as np

def load_pkl(path):
    return pickle.load(open( path, 'rb'))	

def dump_pkl(file, path):
	pickle.dump(file, open( path, 'wb'))	

aschema = ['PM2.5','PM10']
eschema = ['NO2','CO','O3','SO2']


head = ['stationId', 'utc_time']
schema = head + aschema + eschema 


en2ch = load_pkl('meta/en2ch.pkl')
ch2en = load_pkl('meta/ch2en.pkl')
result = []


model = {}
for k, v in ch2en.items():
    model[v] = {}
    for s in schema:
        model[v][s] = np.full(24, np.nan)

def download(day):
    url = 'http://beijingair.sinaapp.com/data/beijing/{}/{}/csv'.format('extra', 
day)  
    r = requests.get(url)
    with open('extra_data/{}_{}.csv'.format('extra', day), 'wb') as f:  
        f.write(r.content)
    url = 'http://beijingair.sinaapp.com/data/beijing/{}/{}/csv'.format('all', 
day)  
    r = requests.get(url)
    with open('extra_data/{}_{}.csv'.format('all', day), 'wb') as f:  
        f.write(r.content)

def crawl():
    start = datetime.date(2013, 12, 4)
    end = datetime.date(2018, 3, 23)
    numdays = (end - start).days 
    date_list = [(end - datetime.timedelta(days=x)).strftime('%Y%m%d')\
        for x in range(0, numdays)]
    with Pool(processes=8) as p:
        p.map(download, date_list)


def local2utc(date):
    year = int(date[0:4])
    month = int(date[4:6])
    day = int(date[6:8])
    hour = int(date[8:10])
    d = datetime.datetime(year, month, day, hour)
    d = d - datetime.timedelta(seconds=8 * 60* 60)
    d = d.strftime('%Y-%m-%d %H:%M:%S')
    return d


def porcessing(day):
    e_path = 'extra_data/extra_{}.csv'.format(day)
    a_path = 'extra_data/all_{}.csv'.format(day)
    print(a_path, e_path)
    current = model
    try :
        a_df = pd.read_csv(a_path)
        for ch, en in ch2en.items():
            col = a_df[['hour', 'type', ch]]
            for s in aschema:
                row = col.loc[col['type'] == s]
                row = row[['hour', ch]].values
                for h, v in row:
                    current[en][s][int(h)] = v
    except:
        pass
    try :
        e_df = pd.read_csv(e_path)
        for ch, en in ch2en.items():
            col = e_df[['hour', 'type', ch]]
            for s in eschema:
                row = col.loc[col['type'] == s]
                row = row[['hour', ch]].values
                for h, v in row:
                    current[en][s][int(h)] = v
    except:
        pass
    for ch, en in ch2en.items():
        current[en]['stationId'] = en + '_aq'
        current[en]['utc_time'] = list(map(local2utc, [str(int(day) * 
100 + i) for i in range(24)]))
    
    stations = list(current.keys())
    stations.sort()
    result = pd.DataFrame(columns=schema)
    for s in stations:
        df = pd.DataFrame(data=current[s])
        df = df[schema]
        result = result.append(df)
    result.to_csv('process/{}.csv'.format(day), sep=',', 
index=False)



def run():
    start = datetime.date(2013, 12, 4)
    end = datetime.date(2018, 3, 23)
    numdays = (end - start).days 
    date_list = [(end - datetime.timedelta(days=x)).strftime('%Y%m%d')\
        for x in range(0, numdays)]
    with Pool(processes=8) as p:
        p.map(porcessing, date_list) 

# print(model)
    
if __name__ == "__main__":
   ## crawl()
   ## run()
    df = pd.read_csv("data/train.csv")
    df = df.sort_values(by  = head)
    print(len(df))
    df = df.loc[df['stationId'] != 'stationId']
    print(len(df))
    df.to_csv("data/train.csv", sep=',',index=False)
    print(df)
