#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Compute reputation and optimism of nodes of a directed signed network.

Implement the method described in
Zhaoming Wu, Charu C. Aggarwal, and Jimeng Sun. 2016.
The Troll-Trust Model for Ranking in Signed Networks.
In Proceedings of the Ninth ACM International Conference on Web Search and Data
Mining (WSDM '16). pp 447-456. http://dx.doi.org/10.1145/2835776.2835816.
"""
from collections import defaultdict
import random
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier
import numpy as np
from timeit import default_timer as clock
warnings.filterwarnings('ignore', r'invalid value encountered in true_divide',
                        RuntimeWarning)


class NodesRanker(BaseEstimator, TransformerMixin):

    """Assign a rank to each node based on its trustworthiness."""

    def __init__(self, beta=0.5, lambda_1=1, tol=1e-11, n_iters=200,
                 autotune_budget=0):
        """Initialize the parameters.

        beta: default value of trustworthiness
        lambda_1: edge weight multiplier
        tol: no update when change get smaller than tol
        n_iters: maximum number of iteration
        autotune_budget: number of couples (β, λ) to try by cross-validation on
        the train set
        """
        self.beta = beta
        self.lambda_1 = lambda_1
        self.tol = tol
        self.n_iters = n_iters
        self.autotune_budget = autotune_budget

    def fit(self, E, N):
        """compute nodes ranking, reputation and optimism.

        E is a dictionary of edge weights (meaning it contains the label used
        later in prediction) and N is the total number of nodes in the whole
        graph.
        """
        # TODO: to be closer to scikit-learn philosophy, X should be a sparse
        # adjacency matrix and should be extracted out of it.
        self.Etrain = E
        self.N = N
        Esplit, params = [], []
        logreg = None
        self.time_taken = 0
        sstart = clock()
        if self.autotune_budget > 0:
            Esplit = [NodesRanker.split_edges(E, .8) for _ in range(5)]
            params = [(random.uniform(0, 1), random.uniform(.1, 2))
                      for _ in range(self.autotune_budget)]
            params = [(b, l) for b in np.linspace(.1, .9, 7)
                      for l in np.linspace(.3, 1.1, 3)]
            logreg = SGDClassifier(loss='log', n_iter=2, class_weight={0: 1.4, 1: 1},
                                   warm_start=True, average=True)
        perfs = []
        for beta, lambda_1 in params:
            self.beta, self.lambda_1 = beta, lambda_1
            perf = 0
            for split in Esplit:
                self.Etrain, Etest = split
                self._compute_rpi()
                self._compute_rep_opt()
                Xtrain, ytrain = self.transform(self.Etrain)
                Xtest, ytest = self.transform(Etest)
                logreg.fit(Xtrain, ytrain)
                perf += np.sum(logreg.predict(Xtest) == ytest)
            perfs.append(perf)
        if perfs:
            self.beta, self.lambda_1 = params[np.argmax(perfs)]
            print(self.beta, self.lambda_1)
        self.Etrain = E
        self.time_taken += clock() - sstart
        self._compute_rpi()
        self._compute_rep_opt()

    def transform(self, E):
        """Return a matrix of edge features."""
        return self._format_feature(E)

    def _compute_rpi(self):
        E = self.Etrain
        N = self.N
        G = {u: set() for u in range(N)}
        for v, u in E:
            G[u].add(v)
        cst = np.log(self.beta/(1-self.beta))
        incoming = [[] if len(G[i]) == 0 else sorted(G[i]) for i in range(N)]
        wincoming = [np.array([self.lambda_1 * E[(j, i)] for j in ins])
                     for i, ins in enumerate(incoming)]
        pi = self.beta*np.ones(N)
        sstart = clock()
        nb_iter, eps = 0, 1
        while eps > self.tol and nb_iter < self.n_iters:
            next_pi = np.zeros_like(pi)
            for i in range(N):
                if not incoming[i]:
                    continue
                pij = pi[incoming[i]]
                ratio = pij/(1 + np.exp(-cst - wincoming[i]))
                opp_prod = np.exp(np.log(1-pij).sum())
                next_pi[i] = (ratio.sum() + self.beta*opp_prod)/(pij.sum() + opp_prod)
            eps = (np.abs(next_pi - pi)/pi).mean()
            pi = next_pi.copy()
            nb_iter += 1
        self.time_taken += clock() - sstart
        self.rpi = pi

    def _compute_rep_opt(self):
        rep_plus, rep_minus = defaultdict(int), defaultdict(int)
        opt_plus, opt_minus = defaultdict(int), defaultdict(int)
        rpi = self.rpi
        sstart = clock()
        for (u, v), pos in self.Etrain.items():
            if pos > 0:
                opt_plus[u] += rpi[v]
                rep_plus[v] += rpi[u]
            else:
                opt_minus[u] += rpi[v]
                rep_minus[v] += rpi[u]

        rep, opt = np.zeros_like(rpi),  np.zeros_like(rpi)
        for u in range(rpi.size):
            rdenom = (rep_plus[u] + rep_minus[u])
            rep[u] = .5 if rdenom == 0 else (rep_plus[u] - rep_minus[u])/rdenom
            odenom = (opt_plus[u] + opt_minus[u])
            opt[u] = .5 if odenom == 0 else (opt_plus[u] - opt_minus[u])/odenom
        self.time_taken += clock() - sstart
        self.reputation = rep
        self.optimism = opt

    def _format_feature(self, E):
        X, y = np.zeros((len(E), 4)), np.zeros(len(E), dtype='int8')
        for i, ((u, v), pos) in enumerate(E.items()):
            X[i, :] = [self.reputation[u], self.reputation[v],
                       self.optimism[u], self.optimism[v]]
            y[i] = int((pos+1)//2)
        return X, y

    @staticmethod
    def split_edges(E, train_fraction):
        """Split `E` in (train_fraction, 1-train_fraction) parts."""
        Etrain, Etest = {}, {}
        for e, s in E.items():
            if random.random() < train_fraction:
                Etrain[e] = 1 if s > 0 else -1
            else:
                Etest[e] = 1 if s > 0 else -1
        return (Etrain, Etest)

if __name__ == '__main__':
    # pylint: disable=C0103
    from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, f1_score
    import pack_graph as pg
    import sys

    if len(sys.argv) == 1:
        datafile = 'directed_wik.pack'
    else:
        datafile = sys.argv[1]
    G, E = pg.load_directed_signed_graph(datafile)
    logreg = SGDClassifier(loss='log', n_iter=5, class_weight={0: 1.4, 1: 1},
                           warm_start=True, average=True)
    nrk = NodesRanker(autotune_budget=0)

    k = 5
    res = np.zeros((k, 5))
    for i in range(k):
        Etrain, Etest = NodesRanker.split_edges(E, .9)

        nrk.fit(Etrain, len(G))
        print(nrk.time_taken)
        Xtrain, ytrain = nrk.transform(Etrain)
        Xtest, ytest = nrk.transform(Etest)
        gold = ytest
        start = clock()
        logreg.fit(Xtrain, ytrain)
        end = clock() - start + nrk.time_taken
        pred = logreg.predict(Xtest)
        C = confusion_matrix(gold, pred)
        fp, tn = C[0, 1], C[0, 0]
        res[i, :] = [accuracy_score(gold, pred),
                     f1_score(gold, pred, average='weighted', pos_label=None),
                     matthews_corrcoef(gold, pred), fp/(fp+tn), end]
        print(res[i, :])
    print(res.mean(0), res.std(0))
