#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Learn a depth 2, two features decision tree by using a
DecisionTreeClassifier for each leaf."""
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdhocDecisionTree(object):
    def __init__(self, negative_weight=1.4, troll_first=True):
        self.order = (0, 1) if troll_first else (1, 0)
        cw = {0: negative_weight, 1: 1}
        self.top =  DecisionTreeClassifier(criterion='gini', max_features=None,
                                           max_depth=1, class_weight=cw)
        self.left_tree =  DecisionTreeClassifier(criterion='gini', max_depth=1,
                                                 max_features=None, class_weight=cw)
        self.right_tree =  DecisionTreeClassifier(criterion='gini', max_depth=1,
                                                  max_features=None, class_weight=cw)

    def fit(self, features, labels):
        self.top.fit(features[:, self.order[0]][:,np.newaxis], labels)
        tt = self.top.tree_.threshold[0]
        l, r = features[:, self.order[0]] <= tt, features[:, self.order[0]] > tt
        self.left_tree.fit(features[l, self.order[1]][:,np.newaxis], labels[l])
        tl = self.left_tree.tree_.threshold[0]
        decisionl = [int(c[1] > c[0])
                     for c in self.left_tree.tree_.value.reshape((3,2))[[1,2],:]]
        self.right_tree.fit(features[r, self.order[1]][:,np.newaxis], labels[r])
        tr = self.right_tree.tree_.threshold[0]
        decisionr = [int(c[1] > c[0])
                     for c in self.right_tree.tree_.value.reshape((3,2))[[1,2],:]]
        self.threshold = [tt, tl, tr]
        self.decision = decisionl + decisionr

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


if __name__ == '__main__':
    # pylint: disable=C0103
    from timeit import default_timer as clock
    # with np.load('sla_dt_feat.npz') as f:
    #     Xa, ya = f['Xa'], f['ya']
    # n = Xa.shape[0]
    # train_set = np.arange(n)
    # test_set = np.arange(n)
    import LillePrediction as llp
    graph = llp.LillePrediction(use_triads=False)
    graph.load_data(llp.lp.DATASETS.Wikipedia, balanced=True)
    Esign=graph.select_train_set(sampling=lambda d: int(.9*d))
    print(100*len(Esign)/len(graph.E))
    Xl, yl, train_set, test_set = graph.compute_features()
    Xa, ya = np.array(Xl)[:,15:17], np.array(yl)

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    from sklearn.metrics import confusion_matrix
    dt = DecisionTreeClassifier(criterion='gini', max_features=None,
                                max_depth=2, class_weight={0: 1.4, 1: 1})
    mydtt = AdhocDecisionTree(troll_first=True)
    mydt = AdhocDecisionTree(troll_first=False)
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
    print([accuracy_score(gold, pred), f1_score(gold, pred, average='weighted', pos_label=None),
           matthews_corrcoef(gold, pred), fp/(fp+tn), end])

    start = clock()
    mydt.fit(Xa[train_set, :], ya[train_set])
    pred = mydt.predict(Xa[test_set, :])
    end = clock() - start
    C = confusion_matrix(gold, pred)
    fp, tn = C[0, 1], C[0, 0]
    print('my tree with pleas first:')
    print(mydt.threshold, mydt.decision)
    print([accuracy_score(gold, pred), f1_score(gold, pred, average='weighted', pos_label=None),
           matthews_corrcoef(gold, pred), fp/(fp+tn), end])

    start = clock()
    mydtt.fit(Xa[train_set, :], ya[train_set])
    pred = mydtt.predict(Xa[test_set, :])
    end = clock() - start
    C = confusion_matrix(gold, pred)
    fp, tn = C[0, 1], C[0, 0]
    print('my tree with troll first:')
    print(mydtt.threshold, mydt.decision)
    print([accuracy_score(gold, pred), f1_score(gold, pred, average='weighted', pos_label=None),
           matthews_corrcoef(gold, pred), fp/(fp+tn), end])
