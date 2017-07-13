#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import SGDClassifier

from adhoc_DT import AdhocDecisionTree


class QuadrantClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, negative_weight=1.4, lambdas=[.5, .5, .5], troll_first=True, 
                 sub_strategy='majority', inner_classifier=None, Fabio_bias=False):
        assert sub_strategy in ['majority', 'perceptron'], '{} is not a valid sub_strategy'
        self.lambdas = lambdas        
        default = SGDClassifier(loss="perceptron", eta0=1, n_iter=4,
                                class_weight={0: negative_weight, 1: 1},
                                learning_rate="constant", penalty=None, average=True)
        self.inner_classifier = inner_classifier if inner_classifier else default
        self.negative_weight = negative_weight
        self.sub_strategy = sub_strategy
        self.troll_first = troll_first
        self.Fabio_bias = Fabio_bias
        if self.sub_strategy == 'perceptron':
            self.pa2 = clone(self.inner_classifier)
            self.pa4 = clone(self.inner_classifier)
            self.pa1 = clone(self.inner_classifier)
            self.pa3 = clone(self.inner_classifier)

    def fit(self, X, y):
        assert X.shape[1] == 2, 'only two features'
        assert 0 <= X.min() and X.max() <= 1, 'features should be [0,1] ratio'
        if self.lambdas is None:
            tmpdt = AdhocDecisionTree(self.negative_weight, self.troll_first)
            tmpdt.fit(X, y)
            self.lambda1, self.lambda2, self.lambda3 = tmpdt.threshold
        else:
            self.lambda1, self.lambda2, self.lambda3 = self.lambdas
        ff, sf = (0, 1) if self.troll_first else (1, 0)
        non_troll_unpleasant = np.logical_and(X[:, ff] <= self.lambda1, X[:, sf] > self.lambda2)
        troll_pleasant = np.logical_and(X[:, ff] > self.lambda1, X[:, sf] <= self.lambda3)
        non_troll_pleasant = np.logical_and(X[:, ff] <= self.lambda1, X[:, sf] <= self.lambda2)
        troll_unpleasant = np.logical_and(X[:, ff] > self.lambda1, X[:, sf] > self.lambda3)
        if self.sub_strategy == 'majority':
            self._predict2 = lambda X: np.argmax(np.bincount(y[non_troll_unpleasant]))
            self._predict4 = lambda X: np.argmax(np.bincount(y[troll_pleasant]))
            self._predict1 = lambda X: np.argmax(np.bincount(y[troll_unpleasant]))
            self._predict3 = lambda X: np.argmax(np.bincount(y[non_troll_pleasant]))
        if self.sub_strategy == 'perceptron':
            self.pa2.fit(X[non_troll_unpleasant, :], y[non_troll_unpleasant])
            self._predict2 = self.pa2.predict
            self.pa4.fit(X[troll_pleasant, :], y[troll_pleasant])
            self._predict4 = self.pa4.predict
            self.pa1.fit(X[troll_unpleasant, :], y[troll_unpleasant])
            self._predict1 = self.pa1.predict
            self.pa3.fit(X[non_troll_pleasant, :], y[non_troll_pleasant])
            self._predict3 = self.pa3.predict

    def predict(self, X):
        assert X.shape[1] == 2, 'only two features'
        assert 0 <= X.min() and X.max() <= 1, 'features should be [0,1] ratio'
        ff, sf = (0, 1) if self.troll_first else (1, 0)
        non_troll_unpleasant = np.logical_and(X[:, ff] <= self.lambda1, X[:, sf] > self.lambda2)
        troll_pleasant = np.logical_and(X[:, ff] > self.lambda1, X[:, sf] <= self.lambda3)
        non_troll_pleasant = np.logical_and(X[:, ff] <= self.lambda1, X[:, sf] <= self.lambda2)
        troll_unpleasant = np.logical_and(X[:, ff] > self.lambda1, X[:, sf] > self.lambda3)
        pred = np.zeros(X.shape[0])
        pred[non_troll_unpleasant] = self._predict2(X[non_troll_unpleasant, :])
        pred[non_troll_pleasant] = 1 if self.Fabio_bias else self._predict3(X[non_troll_pleasant, :])
        pred[troll_pleasant] = self._predict4(X[troll_pleasant, :])
        pred[troll_unpleasant] = 0 if self.Fabio_bias else self._predict1(X[troll_unpleasant, :])
        return pred

if __name__ == '__main__':
    # pylint: disable=C0103
    import doctest
    doctest.testmod()
