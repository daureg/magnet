#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Learn a depth 2, two features decision tree by brute force."""
import numpy as np
from multiprocessing import Pool

CENTERS = {(1, 0): {False: [.28, .45, .14],
                    True: [.38, .58, .02]},
           (0, 1): {False: [.305, .505, .12],
                    True: [.29, .09, .67]}}

class AdhocDecisionTree(object):
    def __init__(self, negative_weight=1.4, troll_first=True, is_epinion=False,
                 num_threads=16):
        self.order = (0, 1) if troll_first else (1, 0)
        self.centers = CENTERS[self.order][is_epinion]
        self.neg_weight = negative_weight
        self.pool = Pool(num_threads)

    def fit(self, features, labels):
        n = features.shape[0]
        w = self.neg_weight
        steps = 16*8
        gap = .4
        trange = np.linspace(max(0, self.centers[0]-gap), 
                             min(1, self.centers[0]+gap), steps)
        tt, l, r = find_split(self.pool, features[:, self.order[0]], labels, np.arange(n), trange, w)
        trange = np.linspace(max(0, self.centers[1]-gap), 
                             min(1, self.centers[1]+gap), steps)
        tl, ll, lr = find_split(self.pool, features[:, self.order[1]], labels, l, trange, w)
        trange = np.linspace(max(0, self.centers[2]-gap), 
                             min(1, self.centers[2]+gap), steps)
        tr, rl, rr = find_split(self.pool, features[:, self.order[1]], labels, r, trange, w)
        self.threshold = [tt, tl, tr]
        self.decision = [leaf_repartition(n, np.argwhere(sub).ravel(), labels[first], w)[0]
                         for sub, first in zip([ll, lr, rl, rr], [l, l, r, r])]

    def predict(self, features):
        threshold = self.threshold
        decision = self.decision
        feature_order = self.order
        l, r = features[:, feature_order[0]] <= threshold[0], features[:, feature_order[0]] >= threshold[0]
        ll, lr = features[l, feature_order[1]] <= threshold[1], features[l, feature_order[1]] >= threshold[1]
        rl, rr = features[r, feature_order[1]] <= threshold[2], features[r, feature_order[1]] >= threshold[2]
        pred = np.zeros(features.shape[0])
        pred.flat[np.flatnonzero(l)[ll]] = decision[0]
        pred.flat[np.flatnonzero(l)[lr]] = decision[1]
        pred.flat[np.flatnonzero(r)[rl]] = decision[2]
        pred.flat[np.flatnonzero(r)[rr]] = decision[3]
        return pred


def inner_split(Xa, ya, samples, w, t):
    left, right=Xa[samples]<=t, Xa[samples]>t
    sl = left.copy().astype(float)
    sl[np.logical_and(ya[samples]<.5, left)] *= w
    sr = right.copy().astype(float)
    sr[np.logical_and(ya[samples]<.5, right)] *= w
    gl = sl.sum()*compute_gini(ya[samples][left], ya[samples][left]>.5, w)
    gr = sr.sum()*compute_gini(ya[samples][right], ya[samples][right]>.5, w)
    return gl+gr
from itertools import repeat
def find_split(pool, Xa, ya, samples, trange, w):
    n = len(trange)
    res = pool.starmap(inner_split,
                        zip(repeat(Xa, n), repeat(ya, n), repeat(samples, n),
                            repeat(w, n), trange), 3)
    # for t in trange:
    #     left, right=Xa[samples]<=t, Xa[samples]>t
    #     sl = left.copy().astype(float)
    #     sl[np.logical_and(ya[samples]<.5, left)] *= w
    #     sr = right.copy().astype(float)
    #     sr[np.logical_and(ya[samples]<.5, right)] *= w
    #     gl = sl.sum()*compute_gini(ya[samples][left], ya[samples][left]>.5, w)
    #     gr = sr.sum()*compute_gini(ya[samples][right], ya[samples][right]>.5, w)
    #     res.append(gl+gr)
    threshold = trange[np.argmin(res)]
    left = Xa[samples]<=threshold
    right = Xa[samples]>threshold
    return threshold, left, right


def leaf_repartition(n, samples, labels, w):    
    frac = samples.size/n
    pos = labels[samples].sum()/samples.size
    neg = (samples.size - labels[samples].sum())/samples.size
    wn = neg*w/(w*neg+pos)
    wp = pos*1/(w*neg+pos)
    return int(wp > wn), wn, wp


def compute_gini(labels, pos, w):
    if labels.size == 0:
        return 0
    pos = labels[pos].size/labels.size
    neg = 1-pos
    wn = neg*w/(w*neg+pos)
    wp = pos*1/(w*neg+pos)
    return 2*wn*wp


if __name__ == '__main__':
    # pylint: disable=C0103
    from timeit import default_timer as clock
    # with np.load('sla_dt_feat.npz') as f:
    #     Xa, ya = f['Xa'], f['ya']
    import LillePrediction as llp
    graph = llp.LillePrediction(use_triads=False)
    graph.load_data(llp.lp.DATASETS.Wikipedia, balanced=True)
    Esign=graph.select_train_set(sampling=lambda d: int(.2*d))
    print(100*len(Esign)/len(graph.E))
    Xl, yl, train_set, test_set = graph.compute_features()
    Xa, ya = np.array(Xl)[:,15:17], np.array(yl)

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    from sklearn.metrics import confusion_matrix
    dt = DecisionTreeClassifier(criterion='gini', max_features=None,
                                max_depth=2, class_weight={0: 1.4, 1: 1})
    mydtt = AdhocDecisionTree(troll_first=True, is_epinion=False)
    mydt = AdhocDecisionTree(troll_first=False, is_epinion=False)
    gold = ya[test_set]
    def tree_analysis(t):
        var = t.feature[[0,1,4]]
        thr = t.threshold[[0,1,4]]
        cls = [int(c[1] > c[0]) for c in t.value.reshape((7,2))[[2,3,-2,-1], :]]
        return var, thr, cls

    start = clock()
    dt.fit(Xa[train_set, :], ya[train_set])
    pred = dt.predict(Xa[test_set, :])
    end = clock() - start
    C = confusion_matrix(gold, pred)
    fp, tn = C[0, 1], C[0, 0]
    var, thr, cls = tree_analysis(dt.tree_)
    print('SK tree with {} first:'.format('troll' if var[0] == 0 else 'pleas'))
    print(list(thr), list(cls))
    print([accuracy_score(gold, pred), f1_score(gold, pred),
           matthews_corrcoef(gold, pred), fp/(fp+tn), end])

    start = clock()
    mydt.fit(Xa[train_set, :], ya[train_set])
    pred = mydt.predict(Xa[test_set, :])
    end = clock() - start
    C = confusion_matrix(gold, pred)
    fp, tn = C[0, 1], C[0, 0]
    print('my tree with pleas first:')
    print(mydt.threshold, mydt.decision)
    print([accuracy_score(gold, pred), f1_score(gold, pred),
           matthews_corrcoef(gold, pred), fp/(fp+tn), end])

    start = clock()
    mydtt.fit(Xa[train_set, :], ya[train_set])
    pred = mydtt.predict(Xa[test_set, :])
    end = clock() - start
    C = confusion_matrix(gold, pred)
    fp, tn = C[0, 1], C[0, 0]
    print('my tree with troll first:')
    print(mydtt.threshold, mydt.decision)
    print([accuracy_score(gold, pred), f1_score(gold, pred),
           matthews_corrcoef(gold, pred), fp/(fp+tn), end])
