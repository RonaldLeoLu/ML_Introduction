import numpy as np
from scipy.optimize import leastsq

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""
def ols(x, y):
    """
    Ordinary Least Square Method
    """
    X = np.column_stack((x, np.ones((x.shape[0],1))))
    W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y.T).flatten()

    w, b = W[:-1], W[-1]

    return w, b



def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw

    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_w variable. You should also use and update the velocity v.     #
    ###########################################################################
    '''
    v_t+1 = rho * v_t - learning_rate * dx
    x_t+1 = x_t + v_t+1
    '''
    rho = config['momentum']
    lr = config['learning_rate']

    v = rho * v - lr * dw
    next_w = w + v
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    config['velocity'] = v

    return next_w, config



def rmsprop(x, dx, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(x))

    next_x = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of x #
    # in the next_x variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################
    '''
    s = decay_rate * s + (1 - decay_rate) * dx * dx
    dx'= learning_rate / np.sqrt(s + epsilon) * dx
    next_x = x - dx'
    '''
    lr = config['learning_rate']
    dr = config['decay_rate']
    eps = config['epsilon']
    cache = config['cache']

    cache = dr * cache + (1 - dr) * dx * dx
    next_x = x - lr / np.sqrt(cache + eps) * dx
    config['cache'] = cache
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_x, config


def adam(x, dx, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 1)

    next_x = None
    ###########################################################################
    # TODO: Implement the Adam update formula, storing the next value of x in #
    # the next_x variable. Don't forget to update the m, v, and t variables   #
    # stored in config.                                                       #
    ###########################################################################
    '''
    m_t = beta1 * m_t-1 + (1 - beta1) * dx
    v_t = beta2 * v_t-1 + (1 - beta2) * dx * dx
    new_mt = m_t / (1 - beta1^(t+1))
    new_vt = v_t / (1 - beta2^(t+1))
    gt = learning_rate * new_mt / np.sqrt(new_vt + epsilon)
    x_t = x_t-1 - gt
    '''
    m = config['m']
    v = config['v']
    t = config['t'] + 1
    lr = config['learning_rate']
    eps = config['epsilon']
    beta1 = config['beta1']
    beta2 = config['beta2']

    m_t = beta1 * m + (1 - beta1) * dx
    v_t = beta2 * v + (1 - beta2) * dx * dx
    new_mt = m_t / (1 - beta1 ** t)
    new_vt = v_t / (1 - beta2 ** t)
    next_x = x - lr * new_mt / np.sqrt(new_vt + eps)

    config['m'] = m_t
    config['v'] = v_t
    config['t'] = t
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_x, config
