from sklearn.model_selection import learning_curve, ShuffleSplit
import numpy as np
import matplotlib.pyplot as plt

def show_learning_curve(estimator, X, y):
    np.random.seed(123)
    samples, trs, tes = learning_curve(estimator, X, y,
        cv=ShuffleSplit(n_splits=5), n_jobs=-1, train_sizes=np.linspace(.1,1,10))

    _ = plt.plot(samples, np.mean(trs,axis=1),'bo-',label='train scores')
    _ = plt.plot(samples,np.mean(tes, axis=1),'ro-', label='validation scores')
    plt.legend(loc='best')

    return plt