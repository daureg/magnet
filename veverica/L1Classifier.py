# vim: set fileencoding=utf-8
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from math import sqrt

class L1Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_iter=9):
        self.k = None
        self.n_iter = n_iter

    def fit(self, X, y):
        assert X.shape[1] == 2, 'only two features'
        assert 0 <= X.min() and X.max() <= 1, 'features should be [0,1] ratio'
        dos, k = 0, 1
        n=y.size
        racine = sqrt(2)
        for i in range(self.n_iter):
            dok = k/racine
            step = (dok-dos)/2
            dc, df = dos + step, dok + step
            kc, kf = racine*dc, racine*df
            acc_c, acc_f = (y==(X[:, 0] < (kc-X[:, 1]))).sum(), (y==(X[:, 0] < (kf-X[:, 1]))).sum()
            dos, k = (dos, kc) if acc_c > acc_f else (dok, kf)
        self.k = k

    def predict(self, X):
        return (X[:, 0] < (self.k-X[:, 1])).astype(int)
