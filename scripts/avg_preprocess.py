import pandas as pd
import numpy as np

def avg_preprocess(file):
    nba = pd.read_csv(file)
    #print('read done')
    data = nba[['姓名','赛季','时间','出场','首发','助攻','篮板']]
    #print('select done')
    data.columns = ['name', 'season', 'time', 'appearence', 'starting', 'assist', 'rebound']
    #print('rename done')
    cond1 = (data['season'] == '16--17')
    #print('condition1 done')
    cond2 = (data['time'] > 25.0)
    dt = data[cond1 & cond2].drop_duplicates()
    #print('condition2 done')
    cond3 = ((dt['starting'] / dt['appearence']) > (60 / 88.0)) & (dt['appearence'] > 60)
    #cond4 = (data['appearence'] > 60)

    return dt[cond3]


