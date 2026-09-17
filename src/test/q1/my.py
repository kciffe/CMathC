# q1_preprocess.py
# 第一问数据预处理：MWR + ROBS + RAD
# 保留完整数据，不包含模型计算

import os
import re
import numpy as np
import pandas as pd

DATA_ROOT = r"D:\8\Desktop\CMathc\data\D题\题目数据及检验\随题数据\第一题"
OUTPUT = './output/q1_dataset.csv'


def get_time(name):
    t = re.search(r'20\d{12}', name).group()
    return pd.to_datetime(t, format='%Y%m%d%H%M%S')


def num(x):
    try:
        if str(x).strip() in ['/////', '//////', '////////', '///', '-', '']:
            return np.nan
        x = float(x)
        return np.nan if x in [9999,99999,999998,999999,-9999] else x
    except:
        return np.nan


def read_robs(path, station):
    rows = []
    t = get_time(path)

    flag = False
    for line in open(path, encoding='utf8', errors='ignore'):
        line = line.strip()

        if line == 'ROBS':
            flag = True
            continue

        if line == 'NNNN':
            break

        if flag:
            p = line.split()
            if len(p) >= 7:
                rows.append([
                    station, t,
                    num(p[0]), num(p[1]), num(p[2]),
                    num(p[3]), num(p[4]),
                    num(p[5]), num(p[6])
                ])

    return pd.DataFrame(rows, columns=[
        'station','time','height',
        'wind_direction','wind_speed',
        'vertical_velocity',
        'horizontal_reliability',
        'vertical_reliability',
        'cn2'
    ])


def read_rad(path, station):
    rows = []
    t = get_time(path)
    beam = None

    for line in open(path, encoding='utf8', errors='ignore'):

        line = line.strip()

        if line.startswith('RAD '):
            beam = line.replace('RAD ','').strip()
            continue

        if line == 'NNNN':
            beam = None
            continue

        if beam:
            p = line.split()
            if len(p) >= 4:
                rows.append([
                    station,t,beam,
                    num(p[0]),num(p[1]),
                    num(p[2]),num(p[3])
                ])

    return pd.DataFrame(rows, columns=[
        'station','time','beam','height',
        'spectral_width','snr',
        'radial_velocity'
    ])


def read_mwr(path, station):

    df = pd.read_csv(
        path,
        sep='\t',
        encoding='gb18030',
        encoding_errors='replace',
        skiprows=[0]
    )

    df['time'] = pd.to_datetime(df['DateTime'])

    heights = [
        c for c in df.columns
        if '(km)' in str(c) and re.match(r'^\d', str(c))
    ]

    rows=[]

    names={
        11:'temperature',
        12:'water_vapor_density',
        13:'relative_humidity',
        14:'liquid_water'
    }

    for _,r in df.iterrows():

        try:
            dtype=int(r['10'])
        except:
            continue

        if dtype not in names:
            continue

        for h in heights:
            z=float(re.findall(r'\d+\.?\d*',h)[0])*1000

            rows.append([
                station,
                r.time,
                z,
                names[dtype],
                num(r[h])
            ])

    out=pd.DataFrame(rows,columns=[
        'station','time','height',
        'variable','value'
    ])

    return out.pivot_table(
        index=['station','time','height'],
        columns='variable',
        values='value'
    ).reset_index()


def process_station(path, station):

    mwr=[]
    robs=[]
    rad=[]

    for root,_,files in os.walk(path):

        for f in files:

            fp=os.path.join(root,f)

            if '微波辐射计' in f:
                mwr.append(read_mwr(fp,station))

            elif 'ROBS' in f:
                robs.append(read_robs(fp,station))

            elif 'RAD' in f:
                rad.append(read_rad(fp,station))

    return (
        pd.concat(mwr,ignore_index=True) if mwr else pd.DataFrame(),
        pd.concat(robs,ignore_index=True) if robs else pd.DataFrame(),
        pd.concat(rad,ignore_index=True) if rad else pd.DataFrame()
    )


mwr=[]
robs=[]
rad=[]

for s in ['a站点','b站点']:

    a,b,c=process_station(
        os.path.join(DATA_ROOT,s),
        s[0]
    )

    mwr.append(a)
    robs.append(b)
    rad.append(c)


mwr=pd.concat(mwr,ignore_index=True)
robs=pd.concat(robs,ignore_index=True)
rad=pd.concat(rad,ignore_index=True)

if mwr.empty or robs.empty or rad.empty:
    raise RuntimeError(
        f'数据读取不完整：mwr={mwr.shape}, robs={robs.shape}, rad={rad.shape}，'
        f'请检查 DATA_ROOT={DATA_ROOT}'
    )


# 保留beam，不提前平均
data=robs.merge(
    rad,
    on=['station','time','height'],
    how='left'
)


# MWR插值到风廓线高度
for col in [
    'temperature',
    'relative_humidity',
    'water_vapor_density',
    'liquid_water'
]:

    if col in mwr.columns:

        data[col]=np.nan

        for (s,t),idx in data.groupby(
            ['station','time']
        ).groups.items():

            mw=mwr[
                (mwr.station==s)&
                (mwr.time==t)
            ].dropna(
                subset=[col]
            )

            if len(mw)>1:

                data.loc[idx,col]=np.interp(
                    data.loc[idx,'height'],
                    mw.height,
                    mw[col]
                )


os.makedirs('./output',exist_ok=True)

mwr.to_csv(
    './output/q1_mwr_clean.csv',
    index=False,
    encoding='utf-8-sig'
)

robs.to_csv(
    './output/q1_robs_clean.csv',
    index=False,
    encoding='utf-8-sig'
)

rad.to_csv(
    './output/q1_rad_clean.csv',
    index=False,
    encoding='utf-8-sig'
)

data.to_csv(
    OUTPUT,
    index=False,
    encoding='utf-8-sig'
)

print(data.shape)
print(data.head())
