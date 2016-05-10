#! /usr/bin/env python
import numpy as np
import LillePrediction as llp
from L1Classifier import L1Classifier
from sklearn.base import BaseEstimator, ClassifierMixin
import lprop_matrix as lm
from collections import defaultdict


class WeightedRule2(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.k = None
        self.A = None

    def fit(self, X, y):
        As = np.hstack(np.unique([np.linspace(.3*x, 10*x, 4) for x in [1,]]))
        best_perf = -1
        denom_troll = X[:, 12] + X[:, 5]
        for A in As:
            denom_both = A*(denom_troll) + X[:, 9] + X[:, 4]
            valid_both = denom_both > 0
            both_feats = (A*X[:, 12] + X[:, 4]) / denom_both
            k_both = find_threshold(both_feats[valid_both], y[valid_both])
            pred = pred_with_threshold(both_feats, k_both, denom_both==0)
            perf = (y==pred).sum()
            if perf > best_perf:
                best_perf, self.k, self.A = perf, k_both, A

    def predict(self, X):
        denom_troll = X[:, 12] + X[:, 5]
        denom_both = self.A*(denom_troll) + X[:, 9] + X[:, 4]
        both_feats = (self.A*X[:, 12] + X[:, 4]) / denom_both
        return pred_with_threshold(both_feats, self.k, denom_both==0)


def pred_with_threshold(sv, t, zero_denom):
    pred = np.sign(t - sv).astype(int)
    unsure = np.logical_and(pred == 0, zero_denom)
    pred[unsure] = 2*(np.random.random(unsure.sum())>.5).astype(int)-1
    return (pred+1)//2


def find_threshold(feats, ya, mcc=False):
    N, P = np.bincount(ya)
    rorder = np.argsort(feats)
    size = np.arange(ya.size)+1
    positive = np.cumsum(ya[rorder]).astype(np.double)
    tp = positive
    fp = size - positive
    fn = P - positive
    tn = N - fp

    if mcc:
        measure = (tp*tn - fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    else:
        measure = (tp+tn)/ya.size
    return feats[rorder][np.argmax(measure)]


def init_from_l1_fixed(X, test_set, idx2edge):
    """Provided a more informed initial solution in low regime.

    Make a first estimation with the rule t(i) + u(j) < 1 and the obtained sign
    on the test set to provide initial value of y and estimate p and q as if
    all signs were observed.
    """
    pred = X[test_set, 15:17].sum(1) < 1
    test_set_set = set(test_set)
    test_set_to_pred = {v: i for i, v in enumerate(test_set)}
    y1 = {}
    douth_p, douth_m = defaultdict(int), defaultdict(int)
    dinh_p, dinh_m = defaultdict(int), defaultdict(int)
    for i, (u, v) in idx2edge.items():
        if i in test_set_set:
            pidx = test_set_to_pred[i]
            y1[i] = pred[pidx]
        else:
            y1[i] = int(graph.E[(u, v)])
        (douth_p if y1[i] > 0 else douth_m)[u] += 1
        (dinh_p if y1[i] > 0 else dinh_m)[v] += 1
    douth_p = np.array([douth_p[u] for u in range(n)], dtype=np.uint)
    douth_m = np.array([douth_m[u] for u in range(n)], dtype=np.uint)
    dinh_p = np.array([dinh_p[u] for u in range(n)], dtype=np.uint)
    dinh_m = np.array([dinh_m[u] for u in range(n)], dtype=np.uint)
    douth = douth_p + douth_m
    dinh = dinh_p + dinh_m
    # Using 0.5 instead of random was experimentaly not so good
    p1, q1 = np.random.uniform(.3, 1, n), np.random.uniform(.3, 1, n)
    pdouth = douth > 0
    p1[pdouth] = douth_p[pdouth]/douth[pdouth]
    pdinh = dinh > 0
    q1[pdinh] = dinh_p[pdinh]/dinh[pdinh]
    y1arr = np.array([y1[u] for u in range(len(y1))], dtype=float)
    return np.hstack((p1, q1, y1arr))

if __name__ == '__main__':
    # pylint: disable=C0103
    import time
    import socket
    import argparse
    part = int(socket.gethostname()[-1])-1
    num_threads = 16

    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Which data to use",
                        choices={'wik', 'sla', 'epi', 'kiw'})
    # parser.add_argument("-b", "--balanced", action='store_true',
    #                     help="Should there be 50/50 +/- edges")
    # parser.add_argument("-a", "--active", action='store_true',
    #                     help="Use active sampling strategy")
    parser.add_argument("-n", "--nrep", help="number of repetition", type=int,
                        default=4)
    args = parser.parse_args()
    pref = args.data
    num_rep = args.nrep
    balanced = False  # args.balanced

    graph = llp.LillePrediction(use_triads=False)
    graph.load_data(pref, balanced)
    n = graph.order
    dicho = L1Classifier()
    ar2 = WeightedRule2()
    class_weight = {0: 1.4, 1: 1}
    olr = llp.SGDClassifier(loss="log", learning_rate="optimal", penalty="l2",
                            average=True, n_iter=4, n_jobs=num_threads,
                            class_weight=class_weight)
    if balanced:
        pref += '_bal'

    start = (int(time.time()-(2015-1970)*365.25*24*60*60))//60

    batch = [{'batch': v/100} for v in np.linspace(5, 100, 11).astype(int)]
    batch = [{'batch': v} for v in [.025, .05, .075, .1, .15, .25, .5, .75, .9]]

    data = lm.sio.loadmat('{}_gprime.mat'.format(pref))
    P, sorted_edges = data['P'], data['sorted_edges']
    ya = sorted_edges[:, 2]
    m = sorted_edges.shape[0]

    fres = [[] for _ in range(9)]
    res_file = '{}_{}_{}'.format(pref, start, part+1)
    params_file = '_params_' + res_file
    for params in batch:
        only_troll_fixed, only_troll_learned = [], []
        both_fixed, both_learned = [], []
        l1_learned, l1_fixed = [], []
        logreg = []
        tweaked_r2 = []
        lpmin = []
        for _ in range(num_rep):
            es = graph.select_train_set(**params)
            idx2edge = {i: e for e, i in graph.edge_order.items()}
            Xl, yl, train_set, test_set = graph.compute_features()
            X, ya = np.array(Xl), np.array(yl)
            if len(test_set) < 10:
                test_set = train_set
            gold = ya[test_set]
            pp = None

            pred_function = graph.train(dicho, X[train_set, 15:17], ya[train_set])
            res = graph.test_and_evaluate(pred_function, X[test_set, 15:17], gold, pp)
            l1_learned.append(res)
            frac = len(train_set)/len(graph.E)
            with open(params_file, 'a') as f:
                f.write('{}_l1c\t{:.3f}\t{:.6f}\n'.format(pref, frac, dicho.k))

            pred_function = graph.train(ar2, X[train_set, :], ya[train_set])
            res = graph.test_and_evaluate(pred_function, X[test_set, :], gold, pp)
            tweaked_r2.append(res)
            with open(params_file, 'a') as f:
                f.write('{}_ar2\t{:.3f}\t{:.6f}\t{:.6f}\n'.format(pref, frac, ar2.k, ar2.A))

            pred_function = graph.train(lambda features: features[:, 0] + features[:, 1] < 1)
            res = graph.test_and_evaluate(pred_function, X[test_set, 15:17], gold, pp)
            l1_fixed.append(res)

            denom_troll = X[:, 12] + X[:, 5]
            valid_denom = denom_troll > 0
            tmp_train = np.zeros_like(valid_denom, dtype=bool)
            tmp_train[train_set] = True
            valid_train_denom = np.logical_and(valid_denom, tmp_train)
            troll_feats = X[:, 12] / denom_troll
            k_troll = find_threshold(troll_feats[valid_train_denom], ya[valid_train_denom])

            pred_function = graph.train(lambda features:
                                        pred_with_threshold(features, 0.5, denom_troll[test_set]==0))
            res = graph.test_and_evaluate(pred_function, troll_feats[test_set], gold, pp)
            only_troll_fixed.append(res)
            with open(params_file, 'a') as f:
                f.write('{}_troll\t{:.3f}\t{:.6f}\n'.format(pref, only_troll_fixed[-1][-1],
                                                            find_threshold(troll_feats[valid_train_denom], ya[valid_train_denom])))
            pred_function = graph.train(lambda features:
                                        pred_with_threshold(features,
                                                            find_threshold(troll_feats[valid_train_denom], ya[valid_train_denom]),
                                                            denom_troll[test_set]==0))
            res = graph.test_and_evaluate(pred_function, troll_feats[test_set], gold, pp)
            only_troll_learned.append(res)

            denom_both = denom_troll + X[:, 9] + X[:, 4]
            valid_both = denom_both > 0
            valid_train_both = np.logical_and(valid_both, tmp_train)
            both_feats = (X[:, 12] + X[:, 4]) / denom_both

            pred_function = graph.train(lambda features:
                                        pred_with_threshold(features, 0.5, denom_both[test_set]==0))
            res = graph.test_and_evaluate(pred_function, both_feats[test_set], gold, pp)
            both_fixed.append(res)
            with open(params_file, 'a') as f:
                f.write('{}_both\t{:.3f}\t{:.6f}\n'.format(pref, both_fixed[-1][-1],
                                                           find_threshold(both_feats[valid_train_both], ya[valid_train_both])))
            pred_function = graph.train(lambda features:
                                        pred_with_threshold(features,
                                                            find_threshold(both_feats[valid_train_both], ya[valid_train_both]),
                                                            denom_both[test_set]==0))
            res = graph.test_and_evaluate(pred_function, both_feats[test_set], gold, pp)
            both_learned.append(res)

            pred_function = graph.train(olr, X[train_set, 15:17], ya[train_set])
            res = graph.test_and_evaluate(pred_function, X[test_set, 15:17], gold, pp)
            logreg.append(res)
            log_weight = str(list(olr.coef_[0])+list(olr.intercept_))
            with open(params_file, 'a') as f:
                f.write('{}_logreg\t{:.3f}\t{}\n'.format(pref, frac, log_weight))

            sstart = llp.lp.clock()
            feats = lm._train(P, train_set, ya[train_set], (m, n))
            k_star = -find_threshold(-feats[train_set], ya[train_set])
            pred = feats[test_set] > k_star
            time_elapsed = llp.lp.clock() - sstart

            C = llp.lp.confusion_matrix(gold, pred)
            fp, tn = C[0, 1], C[0, 0]
            acc, fpr, f1, mcc = [llp.lp.accuracy_score(gold, pred), fp/(fp+tn),
                                 llp.lp.f1_score(gold, pred, average='weighted', pos_label=None),
                                 llp.lp.matthews_corrcoef(gold, pred)]
            lpmin.append([acc, f1, mcc, fpr, time_elapsed, frac])

        fres[0].append(only_troll_fixed)
        fres[1].append(both_fixed)
        fres[2].append(l1_fixed)
        fres[3].append(only_troll_learned)
        fres[4].append(both_learned)
        fres[5].append(l1_learned)
        fres[6].append(logreg)
        fres[7].append(tweaked_r2)
        fres[8].append(lpmin)
    # if args.active:
    #     pref += '_active'
        np.savez_compressed(res_file, res=np.array(fres))
