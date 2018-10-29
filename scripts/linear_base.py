import random
import numpy as np
from .optim import *

class LinearBase:
    def __init__(self, learning_rate=1e-5, W=None, b=None, 
        optimizer='sgd', max_iter=1000, batch_size=None):
        '''
        Basic object for linear models' initialization

        Param:
        --------
        learning_rate: learning rate of gradient descent, default 0.00001
        W: matrix of weights, default None
        b: vector of bias, default None
        optimizer: the optimizer used in this algorithm, default 'sgd'.
                   Available options: 'sgd', 'ols', 'adam', 'rmsprop' or you can define 
                   your own optimizer. More details see optim.py
                   'ols' can be only used in linear regression
        max_iter: the maximum times of iteration
        batch_size: size of batch if using sgd-type optimizer
        '''
        self.lr = learning_rate
        self.W = W
        self.b = b
        self.mi = max_iter
        self.bz = batch_size

        if optimizer == 'adam':
            self.opt = adam
        elif optimizer == 'sgd':
            self.opt = sgd
        elif optimizer == 'rmsprop':
            self.opt = rmsprop
        elif hasattr(optimizer,'__call__') == True:
            self.opt = optimizer
        elif optimizer == 'ols':
            self.opt = 'ols'
        else:
            raise ValueError('Invalid function(name) for this parameter.')

 
    def loss(self, X, y):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def coef(self):
        return self.W.flatten().tolist(), self.b.flatten().tolist()


class LinearRegular:
    def __init__(self, learning_rate=1e-3, W=None, b=None, optimizer='sgd', 
        max_iter=1000, batch_size=None, alpha=None, rho=None):
        '''
        Basic object for linear regression with regularization.

        Param:
        --------
        the same as in LinearBase

        alpha: the hyper parameter of penalty
        rho: the hyper parameter used in elastic net regression
        '''
        self.lr = learning_rate
        self.W = W
        self.b = b
        self.mi = max_iter
        self.bz = batch_size
        self.alpha = alpha
        self.rho = rho

        if optimizer == 'adam':
            self.opt = adam
        elif optimizer == 'sgd':
            self.opt = sgd
        elif optimizer == 'rmsprop':
            self.opt = rmsprop
        elif optimizer == 'lbfgs':
            self.opt = 'lbfgs'
        elif hasattr(optimizer,'__call__') == True:
            self.opt = optimizer
        else:
            raise ValueError('Invalid function(name) for this parameter.')

    def vRegular(self):
        pass

    def dRegular(self):
        pass

    def loss(self, X, y):
        grads = {}
        N, M = X.shape

        yhat = X.dot(self.W) + self.b

        loss = 0.5 / N * np.sum(yhat - y.T) + self.vRegular()

        dW = X.T.dot(yhat - y.T) / N + self.dRegular()
        db = np.sum(yhat - y.T) / N

        grads['dW'] = dW
        grads['db'] = db

        return loss, grads

    def fit(self, X, y):
        N, M = X.shape
        if len(y.shape) == 1:
            y = y.reshape((1,-1))
        # Initailization
        if self.W is None:
            self.W = np.ones((M, 1))
        if self.b is None:
            self.b = 0

        self.scores = []

        epoches = 0
        config = {'learning_rate':self.lr}
        loss0 = 0
        loss1 = 1e5
        epsilon = 1e-2

        while (abs(loss0 - loss1) > 1e-2) and (epoches < self.mi):
            epoches += 1
            loss0 = loss1

            if self.bz is None:
                subx, suby = X, y
            else:
                batch_idx = random.sample(list(range(X.shape[0])), self.bz)
                subx, suby = X[batch_idx,:], y[:,batch_idx]

            #print(subx.shape)
            #print(self.W)

            loss1, grads = self.loss(subx, suby)
            self.scores.append(loss1)


            self.W, cache = self.opt(self.W, grads['dW'], config)
            self.b -= self.lr * grads['db']



    def predict(self, X):
        return (X.dot(self.W) + self.b).flatten()

    def coef(self):
        return self.W.flatten().tolist(), self.b.flatten().tolist()

