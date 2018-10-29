from .linear_base import *
import numpy as np
import random
from .optim import *

class RidgeRegression(LinearRegular):
    def vRegular(self):
        return self.alpha * np.sum(np.square(self.W))

    def dRegular(self):
        return 2 * self.alpha * self.W


class LassoRegression(LinearRegular):
    def vRegular(self):
        return self.alpha * np.sum(np.abs(self.W))

    def dRegular(self):
        dw = self.W.copy()

        dw[dw > 0] = 1
        dw[dw < 0] = -1
        return self.alpha * dw


class ElasticNetRegression(LinearRegular):
    def vRegular(self):
        '''
        alpha * rho * l1 + 0.5 * alpha * (1 - rho) * l2
        '''
        return self.alpha * self.rho * np.sum(np.square(self.W)) + \
        0.5 * self.alpha * (1 - self.rho) * np.sum(np.abs(self.W))

    def dRegular(self):
        dw = self.W.copy()

        dw[dw > 0] = 1
        dw[dw < 0] = -1

        return 2 * self.alpha * self.rho * self.W + \
        0.5 * self.alpha * (1 - self.rho) * dw
