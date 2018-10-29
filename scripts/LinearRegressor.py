from .linear_base import *
import numpy as np
import random
from .optim import *

class LinearRegressor(LinearBase):
    def loss(self, X, y):
        grads = {}
        m,n = X.shape
        yhat = np.dot(X, self.W) + self.b
        #print(yhat.shape)
        z = yhat-y.T
        loss = 0.5 / m * np.sum(np.square(z))

        dW = X.T.dot(z) / m
        db = np.sum(z) / m
        #print('in loss db:',db)
        #print(dW)
        #print('db shape:',db)

        grads['dW'] = dW
        grads['db'] = db

        return loss, grads

    def fit(self, X, y):
        self.scores = []
        if len(y.shape) == 1:
            y = y.reshape((1,-1))

        if self.opt == 'ols':
            self.W, self.b = ols(X, y)

            return self

        if self.W is None:
            self.W = np.ones((X.shape[1],1))
        if self.b is None:
            self.b = 0

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

            loss1, grads = self.loss(subx, suby)
            self.scores.append(loss1)


            self.W, cache = self.opt(self.W, grads['dW'], config)
            self.b -= self.lr * grads['db']

            w,b = self.coef()

        return self


    def predict(self, X):
        pred = X.dot(self.W) + self.b

        return pred.flatten()