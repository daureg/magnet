# vim: set fileencoding=utf-8
from sklearn.base import BaseEstimator, ClassifierMixin
from math import sqrt
import numpy as np


class L1Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_iter=6):
        self.k = None
        self.n_iter = n_iter

    def fit(self, X, y):
        assert X.shape[1] == 2, 'only two features'
        assert 0 <= X.min() and X.max() <= 1, 'features should be [0,1] ratio'
        feats = X[:, 0] + X[:, 1]
        # self.k = gss(lambda k: -matthews_corrcoef(y, (feats < k)), .3, .9, self.n_iter)
        # self.k = gss(lambda k: (y != (feats < k)).sum(), .3, .9, self.n_iter)
        mask = feats > 0.8*feats.mean()
        rdst = feats[mask]
        rorder = np.argsort(rdst)
        self.k = rdst[rorder][np.argmax(np.cumsum(2*y[mask][rorder]-1))]

    def predict(self, X):
        return (X[:, 0] < (self.k-X[:, 1]))


gr = (sqrt(5) - 1) / 2
def gss(f, a, b, n_iter=5):
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    for _ in range(n_iter):
        fc = f(c)
        fd = f(d)
        if fc < fd:
            b = d
            d = c  # fd=fc;fc=f(c)
            c = b - gr * (b - a)
        else:
            a = c
            c = d  # fc=fd;fd=f(d)
            d = a + gr * (b - a)
    return (b + a) / 2


ks = np.linspace(.4, .9, 50)
# @profile
def bench(X, y):
    f = X[:, 0] + X[:, 1]
    for k in ks:
        pred = f < k
        acc = (y == pred).sum()

if __name__ == "__main__":
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    from sklearn.metrics import confusion_matrix
    with np.load('wik_dt_feat.npz') as f:
        Xa, ya = f['Xa'], f['ya']
    # bench(Xa, ya)
    L1 = L1Classifier(6)
    L1.fit(Xa, ya)
    gold = ya
    pred = L1.predict(Xa).astype(int)
    C = confusion_matrix(gold, pred)
    fp, tn = C[0, 1], C[0, 0]
    print([accuracy_score(gold, pred), f1_score(gold, pred, average='weighted', pos_label=None),
           matthews_corrcoef(gold, pred), fp/(fp+tn)])
