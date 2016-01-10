#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""."""
import LinkPrediction as lp
import leskovec as l
from copy import deepcopy

class LillePrediction(lp.LinkPrediction):
    """My implementation of LinkPrediction"""

    def load_data(self, dataset, balanced=False):
        l.rw.read_original_graph(lp.FILENAMES[dataset], directed=True,
                                 balanced=balanced)
        Gfull, E = l.rw.G, l.rw.EDGE_SIGN
        self.order = len(Gfull)
        self.dout, self.din = l.defaultdict(int), l.defaultdict(int)
        for u, v in E:
            self.dout[u] += 1
            self.din[v] += 1
        self.common_nei = {e: Gfull[e[0]].intersection(Gfull[e[1]]) for e in E}
        self.Gout, self.Gin = {}, {}
        for u, v in E:
            l.add_neighbor(u, v, self.Gout)
            l.add_neighbor(v, u, self.Gin)
        self.G = self.Gout
        self.E = E

    def select_train_set(self, **params):
        if 'batch' in params:
            alpha = min(params['batch']*self.order/len(self.E), 1.0)
            self.Esign = l.trolls.select_edges(None, self.E, alpha, 'random')
            return self.Esign
        else:
            alpha = params.get('alpha', 0)
            sf = params.get('sampling')
            Eout = l.trolls.select_edges(self.Gout, self.E, alpha, 'uniform', True, sf)
            Ein = l.trolls.select_edges(self.Gin, self.E, alpha, 'uniform', True, sf)
            directed_edges = deepcopy(Ein)
            directed_edges.update(Eout)
            self.Esign = directed_edges
            return directed_edges

    def compute_global_features(self):
        self.din_plus, self.dout_plus = l.defaultdict(int), l.defaultdict(int)
        self.din_minus, self.dout_minus = l.defaultdict(int), l.defaultdict(int)
        self.compute_in_out_degree(self.Esign)

    def compute_in_out_degree(self, edges):
        for (u, v), sign in edges.items():
            if sign is True:
                self.din_plus[v] += 1
                self.dout_plus[u] += 1
            else:
                self.din_minus[v] += 1
                self.dout_minus[u] += 1

    def online_mode(self, edges):
        # partial update of global feature and all edges including u and v
        self.compute_in_out_degree(edges)
        to_update = set(edges)
        more_update = set()
        if self.with_triads:
            for u, v in to_update:
                for w in self.common_nei[(u, v)]:
                    assert (u, w) or (w, u) in self.E
                    assert (v, w) or (w, v) in self.E
                    more_update.add((u, w) if (u, w) in self.E else (w, u))
                    more_update.add((v, w) if (v, w) in self.E else (w, v))
        print('{} new edges'.format(len(edges)))
        to_update.update(more_update)
        print('{} to update in total'.format(len(to_update)))
        for edge in to_update:
            self.features[self.edge_order[edge], :] = self.compute_one_edge_feature(edge)

        
    def compute_one_edge_feature(self, edge):
        u, v = edge
        known_out = self.dout_plus[u]+self.dout_minus[u]
        known_in = self.din_plus[v]+self.din_minus[v]
        degrees = [self.dout[u], self.din[v], len(self.common_nei[(u, v)]),
                   self.din_plus[u], self.din_minus[v],
                   self.dout_plus[u], self.dout_minus[v],
                   self.din[u], self.dout[v],
                   self.din_plus[v], self.din_minus[u],
                   self.dout_plus[v], self.dout_minus[u],
                   self.dout_minus[u]/self.dout[u], self.din_minus[v]/self.din[v],
                   0 if known_out == 0 else self.dout_minus[u]/known_out,
                   0 if known_in == 0 else self.din_minus[v]/known_in,
                   ]
        triads = 16*[0, ]
        if self.with_triads:
            for w in self.common_nei[(u, v)]:
                for t in l.triads_indices(u, v, w, self.Esign):
                    triads[t] += 1
        return degrees+triads

def tree_prediction(features, cst, troll_first=True):
    if troll_first:
        return features[:, 1] < (cst[2] + (features[:, 0]<cst[0])*(cst[1]-cst[2]))
    return features[:, 0] < (cst[2] + (features[:, 1]<cst[0])*(cst[1]-cst[2]))

if __name__ == '__main__':
    # pylint: disable=C0103
    from math import log, sqrt, ceil
    import numpy as np
    import time
    import persistent as p
    import socket
    import argparse
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    part = int(socket.gethostname()[-1])-1
    num_threads = 16

    data = {'WIK': lp.DATASETS.Wikipedia,
            'EPI': lp.DATASETS.Epinion,
            'SLA': lp.DATASETS.Slashdot}
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Which data to use",
                        choices=data.keys(), default='WIK')
    parser.add_argument("-b", "--balanced", action='store_true',
                        help="Should there be 50/50 +/- edges")
    parser.add_argument("-a", "--active", action='store_true',
                        help="Use active sampling strategy")
    parser.add_argument("-n", "--nrep", help="number of repetition", type=int,
                        default=3)
    args = parser.parse_args()
    pref = args.data
    num_rep = args.nrep

    graph = LillePrediction(use_triads=True)
    graph.load_data(data[pref], args.balanced)
    olr = LogisticRegression(C=5.455, solver='lbfgs', n_jobs=num_threads,
                             warm_start=True)
    llr = LogisticRegression(C=1e-3, solver='lbfgs', n_jobs=num_threads,
                             warm_start=True)
    dt = DecisionTreeClassifier(criterion='gini', max_features=None,
                                max_depth=2, class_weight={0: 1.4, 1: 1})
    sdt = DecisionTreeClassifier(criterion='gini', max_features=None,
                                 max_depth=1, class_weight={0: 1.4, 1: 1})
    if args.balanced:
        pref += '_bal'

    start = (int(time.time()-(2015-1970)*365.25*24*60*60))//60
    feats = list(range(7)) + list(range(17, 33))
    fres = [[] for _ in range(19)]
    active = [{'sampling': lambda d: 1},
              {'sampling': lambda d: 3},
              {'sampling': lambda d: int(.1*d)},
              {'sampling': lambda d: int(.3*d)},
              {'sampling': lambda d: int(.6*d)},
              {'sampling': lambda d: int(ceil(log(d)))},
              {'sampling': lambda d: 1 if d==1 else max(1, int(ceil(log(log(d)))))},
              ]
    n, m = graph.order, len(graph.E)
    logc = 1 if n*log(n) < m else 0.4
    batch = [{'batch': 2},
             {'batch': 4},
             {'batch': 8},
             {'batch': int(logc*log(graph.order))},
             # {'batch': int(sqrt(graph.order))},
            ]
    lambdas = l.lambdas[pref]
    for params in active if args.active else batch:
        full_troll_fixed, left_troll_fixed, right_troll_fixed, simple_troll_fixed = [], [], [], []
        full_pleas_fixed, left_pleas_fixed, right_pleas_fixed, simple_pleas_fixed = [], [], [], []
        full_troll_tuned, left_troll_tuned, right_troll_tuned, simple_troll_tuned = [], [], [], []
        full_learned, troll_learned, pleas_learned = [], [], []
        lesko, logreg = [], []
        allones, randompred = [], []
        for _ in range(num_rep):
            graph.select_train_set(**params)
            Xl, yl, train_set, test_set = graph.compute_features()
            Xa, ya = np.array(Xl), np.array(yl)
            train_feat = np.ix_(train_set, feats)
            test_feat = np.ix_(test_set, feats)
            gold = ya[test_set]

            cst = [.5, .5, .5]
            pred_function = graph.train(lambda features: tree_prediction(features, cst, True))
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold)
            full_troll_fixed.append(res)
            cst = [.5, .5, .0]
            pred_function = graph.train(lambda features: tree_prediction(features, cst, True))
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold)
            left_troll_fixed.append(res)
            cst = [.5, 1, .5]
            pred_function = graph.train(lambda features: tree_prediction(features, cst, True))
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold)
            right_troll_fixed.append(res)
            cst = [.5, 1, .0]
            pred_function = graph.train(lambda features: tree_prediction(features, cst, True))
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold)
            simple_troll_fixed.append(res)

            cst = lambdas.copy()
            pred_function = graph.train(lambda features: tree_prediction(features, cst, True))
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold)
            full_troll_tuned.append(res)
            cst = lambdas.copy(); cst[2] = 0
            pred_function = graph.train(lambda features: tree_prediction(features, cst, True))
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold)
            left_troll_tuned.append(res)
            cst = lambdas.copy(); cst[1] = 1
            pred_function = graph.train(lambda features: tree_prediction(features, cst, True))
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold)
            right_troll_tuned.append(res)
            cst = lambdas.copy(); cst[1] = 1; cst[2] = 0
            pred_function = graph.train(lambda features: tree_prediction(features, cst, True))
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold)
            simple_troll_tuned.append(res)

            cst = [.5, .5, .5]
            pred_function = graph.train(lambda features: tree_prediction(features, cst, False))
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold)
            full_pleas_fixed.append(res)
            cst = [.5, .5, .0]
            pred_function = graph.train(lambda features: tree_prediction(features, cst, False))
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold)
            left_pleas_fixed.append(res)
            cst = [.5, 1, .5]
            pred_function = graph.train(lambda features: tree_prediction(features, cst, False))
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold)
            right_pleas_fixed.append(res)
            cst = [.5, 1, .0]
            pred_function = graph.train(lambda features: tree_prediction(features, cst, False))
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold)
            simple_pleas_fixed.append(res)

            pred_function = graph.train(dt, Xa[train_set, 15:17], ya[train_set])
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold)
            full_learned.append(res)
            pred_function = graph.train(sdt, Xa[train_set, 15:16], ya[train_set])
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:16], gold)
            troll_learned.append(res)
            pred_function = graph.train(sdt, Xa[train_set, 16:17], ya[train_set])
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 16:17], gold)
            pleas_learned.append(res)

            pred_function = graph.train(llr, Xa[train_feat], ya[train_set])
            res = graph.test_and_evaluate(pred_function, Xa[test_feat], gold)
            lesko.append(res)
            pred_function = graph.train(olr, Xa[train_set, 15:17], ya[train_set])
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold)
            logreg.append(res)

            pred_function = graph.train(lambda features: [0]+[1,]*(features.shape[0]-1))
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold)
            allones.append(res)

            pred_function = graph.train(lambda features: np.random.rand(features.shape[0])>.5)
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold)
            randompred.append(res)
        fres[0].append(full_troll_fixed)
        fres[1].append(left_troll_fixed)
        fres[2].append(right_troll_fixed)
        fres[3].append(simple_troll_fixed)
        fres[4].append(full_pleas_fixed)
        fres[5].append(left_pleas_fixed)
        fres[6].append(right_pleas_fixed)
        fres[7].append(simple_pleas_fixed)
        fres[8].append(full_troll_tuned)
        fres[9].append(left_troll_tuned)
        fres[10].append(right_troll_tuned)
        fres[11].append(simple_troll_tuned)
        fres[12].append(full_learned)
        fres[13].append(troll_learned)
        fres[14].append(pleas_learned)
        fres[15].append(lesko)
        fres[16].append(logreg)
        fres[17].append(allones)
        fres[18].append(randompred)

    if args.active:
        pref += '_active'
    p.save_var('{}_{}_{}.my'.format(pref, start, part+1), (None, fres))
