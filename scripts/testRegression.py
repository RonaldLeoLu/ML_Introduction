import os
import numpy as np
import pandas as pd

from .LinearRegressor import LinearRegressor
from .LinearRegular import *

from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet

# load sample data
filepath = Path(os.getcwd()).parent / 'datasets' / 'portland.txt'
portland = pd.read_table(filepath, header=None, names=['Size', 'Bedrooms', 'Price'], sep=',')

# preprocessing
portland['Size'] = (portland['Size'] - portland['Size'].mean()) / portland['Size'].std()
portland['Bedrooms'] = (portland['Bedrooms'] - portland['Bedrooms']) / portland['Bedrooms'].std()

x = portland[['Size','Bedrooms']].values
y = portland['Price'].values / 1000.0

# load sklearn models
SkLinR = LinearRegression()
SkLogR = LogisticRegression()
SkRidg = Ridge(alpha=0.003)
SkLaso = Lasso(alpha=0.03)
SkElas = ElasticNet(alpha=0.03, l1_ratio=0.04)

# load own strcuture models
LROls = LinearRegressor(optimizer='ols')
LRSgd = LinearRegressor(max_iter=2500,batch_size=15,optimizer='sgd',learning_rate=4e-3)
ridge = RidgeRegression(alpha=0.003,learning_rate=8e-2,max_iter=3000)
lasso = LassoRegression(alpha=0.03,learning_rate=7.8e-2,max_iter=3000)
ElasN = ElasticNetRegression(alpha=0.03,rho=0.04,max_iter=3000,learning_rate=1e-2)

def testAccurate(model1, model2, model3=None):
    model1.fit(x, y)
    model2.fit(x, y)

    w0, b0 = model1.coef_, model1.intercept_
    w1, b1 = model2.coef()

    print('From sklearn packages:')
    print('W:', w0, 'b:', b0)
    print('From our codes')
    print('w:', w1, 'b:', b1)

    if model3 is not None:
        model3.fit(x, y)
        w2, b2 = model3.coef()
        #print('From our codes')
        print('w\':', w2, 'b\':', b2)

    return None

if __name__ == '__main__':
    from sys import argv
    script, learner = argv

    if learner == 'LinearRegression':
        testAccurate(SkLinR, LROls, LRSgd)
    elif learner == 'ridge':
        testAccurate(SkRidg, ridge)
    elif learner == 'lasso':
        testAccurate(SkLaso, lasso)
    elif learner == 'elastic':
        testAccurate(SkElas, ElasN)
    else:
        raise ValueError('You may have the wrong name of learners.')
