from utils import * 
from arguments import *
import matplotlib.pyplot as plt

def draw(data, title):
    
    for lk, lv in data.items():
        line, = plt.plot(lv.values(), label=lk)
    plt.show()

schema = ['PM2.5','PM10', 'NO2','CO','O3','SO2']
if __name__ == '__main__':
    args = parse()
    print_args(args)
    en2ch, station_info = load_meta(args.meta_dir)
    data = load_train_data(args)
    model = {}
    for s in schema:
        model[s] = {}
        for l, loc in station_info.items():
            model[s][l] = {}
            for t in range(24):
                model[s][l][t] = [] 

    count = 0
    for g, d in data:
        l = ''
        for k, loc in station_info.items():
            if g in loc:
                l = k
        d['day'], d['hour'] = d['utc_time'].str.split(' ', 1).str
        d = d.groupby('hour')
        for t, sub in d:
            t = int(t.split(":")[0])
            for s in schema:
                mu = np.mean(sub[s].values)
                model[s][l][t].append(mu)
    for sk, sv in model.items():
        draw(sv, sk)
        for lk, lv in sv.items():
           for tk, tv in lv.items():
                mu = np.mean(tv)
                std = np.std(tv) 
                model[sk][lk][tk] = mu
                print(lk, sk, mu, std)
