#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""."""
import LinkPrediction as lp
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from adhoc_DT import AdhocDecisionTree
import leskovec as l
from copy import deepcopy
from collections import defaultdict, Counter
import random
import numpy as np
import persistent as p
import spectral_prediction as sp
import getWH
from treestar import initial_spanning_tree
from pack_graph import load_directed_signed_graph

lambdas_troll = {'EPI': [ 0.3138361 ,  0.78747173,  0.09077966],
                 'SLA': [ 0.32362659,  0.73825876,  0.10942871],
                 'WIK': [ 0.31698022,  0.77553025,  0.10040762],
                 'EPI_bal': [ 0.38082290, 0.91833335, 0.09005866],
                 'SLA_bal': [ 0.39331159, 0.91427743, 0.06415257],
                 'WIK_bal': [ 0.38671307, 0.93037849, 0.06227483],}
lambdas_pleas = {'EPI': [ 0.37241124,  0.66195652,  0.04517692],
                 'SLA': [ 0.38920860,  0.68269397,  0.03158007],
                 'WIK': [ 0.37179092,  0.65985961,  0.04577132],
                 'EPI_bal': [ 0.38974499,  0.91350523,  0.10585644],
                 'SLA_bal': [ 0.42502945,  0.94503844,  0.15831973],
                 'WIK_bal': [ 0.41281646,  0.93713953,  0.01006711],}

class LillePrediction(lp.LinkPrediction):
    """My implementation of LinkPrediction"""

    def load_data(self, dataset, balanced=False, small_wiki=False):
        if small_wiki:
            Gfull, E = p.load_var('small_wiki.my')
        elif balanced:
            l.rw.read_original_graph(lp.FILENAMES[dataset], directed=True,
                                     balanced=balanced)
        # conflicting = set()
        # for (u, v), s in l.rw.EDGE_SIGN.items():
        #     opposite_sign = l.rw.EDGE_SIGN.get((v, u))
        #     if opposite_sign is not None and s != opposite_sign:
        #         conflicting.add(tuple(sorted([u, v])))
        # msg = 'Number of conflicting edges in {}: {}'
        # print(msg.format(dataset, 2*len(conflicting)))
        # for u, v in conflicting:
        #     l.rw.remove_signed_edge(u, v, directed=True)
        #     l.rw.remove_signed_edge(v, u, directed=True)
            Gfull, E = l.rw.G, l.rw.EDGE_SIGN
        else:
            pack_name = 'directed_{}.pack'.format(dataset)
            Gfull, E = load_directed_signed_graph(pack_name)
        root = max(Gfull.items(), key=lambda x: len(x[1]))[0]
        Gbfs, _, _ = initial_spanning_tree(Gfull, root)
        self.lcc = set(Gbfs.keys())
        self.order = len(Gfull)
        self.dout, self.din = defaultdict(int), defaultdict(int)
        for u, v in E:
            self.dout[u] += 1
            self.din[v] += 1
        self.common_nei = {e: Gfull[e[0]].intersection(Gfull[e[1]]) for e in E}
        self.Gout, self.Gin = {}, {}
        self.edge_order, in_lcc = {}, []
        for i, (u, v) in enumerate(sorted(E)):
            self.edge_order[(u, v)] = i
            in_lcc.append(u in self.lcc and v in self.lcc)
            l.add_neighbor(u, v, self.Gout)
            l.add_neighbor(v, u, self.Gin)
        self.reciprocal = {ei: self.edge_order[(e[1], e[0])]
                           for e, ei in self.edge_order.items()
                           if (e[1], e[0]) in E}
        self.in_lcc = np.array(in_lcc, dtype=bool)
        self.Gfull = Gfull
        self.G = self.Gout
        self.E = E

    def select_train_set(self, **params):
        self.out_samples, self.in_samples = defaultdict(int), defaultdict(int)
        if 'batch' in params:
            alpha = min(params['batch']*self.order/len(self.E), 1.0)
            alpha = params['batch']
            # TODO: make select_edges return out_plus_samples and out_minus,
            # in_plus, in_minus samples directly
            self.Esign = l.trolls.select_edges(None, self.E, alpha, 'random')
            self.out_samples.update(Counter((e[0] for e in self.Esign)))
            self.in_samples.update(Counter((e[1] for e in self.Esign)))
            return self.Esign
        else:
            alpha = params.get('alpha', 0)
            sf = params.get('sampling')
            replacement = params.get('replacement', False)
            do_out = params.get('do_out', True)
            do_in = params.get('do_in', True)
            assert do_out or do_in, 'something to sample at least'
            directed_edges = None
            if do_out:
                Eout = l.trolls.select_edges(self.Gout, self.E, alpha, 'uniform',
                                             True, sf, replacement)
                self.out_samples.update(Counter((e[0] for e in Eout)))
                directed_edges = deepcopy(Eout)
            if do_in:
                Ein = l.trolls.select_edges(self.Gin, self.E, alpha, 'uniform',
                                            True, sf, replacement)
                self.in_samples.update(Counter((e[1] for e in Ein)))
                if directed_edges is None:
                    directed_edges = deepcopy(Ein)
                else:
                    directed_edges.update(Ein)
            self.Esign = directed_edges
            return directed_edges

    def compute_global_features(self):
        self.din_plus, self.dout_plus = defaultdict(int), defaultdict(int)
        self.din_minus, self.dout_minus = defaultdict(int), defaultdict(int)
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
        to_update.update(more_update)
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
                   0.4999999 if known_out == 0 else self.dout_minus[u]/known_out,
                   0.4999999 if known_in == 0 else self.din_minus[v]/known_in,
                   ]
        triads = 16*[0, ]
        if self.with_triads:
            for w in self.common_nei[(u, v)]:
                for t in l.triads_indices(u, v, w, self.Esign):
                    triads[t] += 1
        triads.extend([self.out_samples[u], self.in_samples[v]])
        return degrees+triads

def tree_prediction(features, cst, troll_first=True):
    if troll_first:
        return features[:, 1] < (cst[2] + (features[:, 0]<cst[0])*(cst[1]-cst[2]))
    return features[:, 0] < (cst[2] + (features[:, 1]<cst[0])*(cst[1]-cst[2]))


def online_exp(graph, pref, start, part, batch_size=500):
    num_threads = 16
    graph.Esign = dict(random.sample(list(graph.E.items()), 400))
    Xl, yl, train_set, test_set = graph.compute_features()
    Xa, ya = np.array(Xl), np.array(yl)
    seen = len(graph.Esign)
    lambdas = l.lambdas[pref]
    fres = [[] for _ in range(19)]
    olr = LogisticRegression(C=5.455, solver='lbfgs', n_jobs=num_threads,
                             warm_start=True)
    # llr = LogisticRegressionCV(Cs=8, cv=4, solver='lbfgs', n_jobs=num_threads,
    llr = LogisticRegression(C=0.02, solver='lbfgs', n_jobs=num_threads,
                             warm_start=True)
    dt = DecisionTreeClassifier(criterion='gini', max_features=None,
                                max_depth=2, class_weight={0: 1.4, 1: 1})
    tdt = DecisionTreeClassifier(criterion='gini', max_features=None,
                                 max_depth=1, class_weight={0: 1.4, 1: 1})
    pdt = DecisionTreeClassifier(criterion='gini', max_features=None,
                                 max_depth=1, class_weight={0: 1.4, 1: 1})
    olr.fit(Xa[train_set, 15:17], ya[train_set])
    dt.fit(Xa[train_set, 15:17], ya[train_set])
    tdt.fit(Xa[train_set, 15:16], ya[train_set])
    pdt.fit(Xa[train_set, 16:17], ya[train_set])
    feats = list(range(7)) + list(range(17, 33))
    train_feat = np.ix_(train_set, feats)
    llr.fit(Xa[train_feat], ya[train_set])
    idx2edge = {i: e for e, i in graph.edge_order.items()}
    while seen < .8*len(graph.E) - batch_size:
        indices = random.sample(test_set, batch_size)
        new_edges = sorted(indices)
        gold = ya[indices]

        cst = [.5, .5, .5]
        pred = tree_prediction(Xa[new_edges, 15:17], cst, True)
        fres[0].append((pred != gold).sum())
        cst = [.5, .5, .0]
        pred = tree_prediction(Xa[new_edges, 15:17], cst, True)
        fres[1].append((pred != gold).sum())
        cst = [.5, 1, .5]
        pred = tree_prediction(Xa[new_edges, 15:17], cst, True)
        fres[2].append((pred != gold).sum())
        cst = [.5, 1, .0]
        pred = tree_prediction(Xa[new_edges, 15:17], cst, True)
        fres[3].append((pred != gold).sum())

        cst = lambdas.copy()
        pred = tree_prediction(Xa[new_edges, 15:17], cst, True)
        fres[4].append((pred != gold).sum())
        cst = lambdas.copy(); cst[2] = 0
        pred = tree_prediction(Xa[new_edges, 15:17], cst, True)
        fres[5].append((pred != gold).sum())
        cst = lambdas.copy(); cst[1] = 1
        pred = tree_prediction(Xa[new_edges, 15:17], cst, True)
        fres[6].append((pred != gold).sum())
        cst = lambdas.copy(); cst[1] = 1; cst[2] = 0
        pred = tree_prediction(Xa[new_edges, 15:17], cst, True)
        fres[7].append((pred != gold).sum())

        cst = [.5, .5, .5]
        pred = tree_prediction(Xa[new_edges, 15:17], cst, False)
        fres[8].append((pred != gold).sum())
        cst = [.5, .5, .0]
        pred = tree_prediction(Xa[new_edges, 15:17], cst, False)
        fres[9].append((pred != gold).sum())
        cst = [.5, 1, .5]
        pred = tree_prediction(Xa[new_edges, 15:17], cst, False)
        fres[10].append((pred != gold).sum())
        cst = [.5, 1, .0]
        pred = tree_prediction(Xa[new_edges, 15:17], cst, False)
        fres[11].append((pred != gold).sum())

        pred = dt.predict(Xa[new_edges, 15:17])
        fres[12].append((pred != gold).sum())
        pred = tdt.predict(Xa[new_edges, 15:16])
        fres[13].append((pred != gold).sum())
        pred = pdt.predict(Xa[new_edges, 16:17])
        fres[14].append((pred != gold).sum())

        test_feat = np.ix_(new_edges, feats)
        pred = llr.predict(Xa[test_feat])
        fres[15].append((pred != gold).sum())
        pred = olr.predict(Xa[new_edges, 15:17])
        fres[16].append((pred != gold).sum())

        fres[17].append((np.ones(batch_size)!=gold).sum())
        fres[18].append(((np.random.rand(batch_size)>.5)!=gold).sum())

        new_edges = {idx2edge[i]: graph.E[idx2edge[i]] for i in indices}
        graph.online_mode(new_edges)
        Xa = graph.features
        print(seen// batch_size)
        train_set = sorted(train_set + indices)
        indices = set(indices)
        test_set = [i for i in test_set if i not in indices]
        olr.fit(Xa[train_set, 15:17], ya[train_set])
        dt.fit(Xa[train_set, 15:17], ya[train_set])
        tdt.fit(Xa[train_set, 15:16], ya[train_set])
        pdt.fit(Xa[train_set, 16:17], ya[train_set])
        train_feat = np.ix_(train_set, feats)
        llr.fit(Xa[train_feat], ya[train_set])
        seen += len(new_edges)
    return fres

if __name__ == '__main__':
    # pylint: disable=C0103
    from math import log, sqrt, ceil
    import time
    import socket
    import argparse
    part = int(socket.gethostname()[-1])-1
    num_threads = 16

    data = {'WIK': lp.DATASETS.Wikipedia,
            'EPI': lp.DATASETS.Epinion,
            'RFA': lp.DATASETS.Rfa,
            'SLA': lp.DATASETS.Slashdot}
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Which data to use",
                        choices=data.keys(), default='WIK')
    parser.add_argument("-b", "--balanced", action='store_true',
                        help="Should there be 50/50 +/- edges")
    parser.add_argument("-a", "--active", action='store_true',
                        help="Use active sampling strategy")
    parser.add_argument("-o", "--online", type=int,
                        help="set the batch size of online mode")
    parser.add_argument("-n", "--nrep", help="number of repetition", type=int,
                        default=3)
    args = parser.parse_args()
    pref = args.data
    num_rep = args.nrep

    graph = LillePrediction(use_triads=True)
    graph.load_data(data[pref], args.balanced)
    class_weight = {0: 1.4, 1: 1}
    pac = PassiveAggressiveClassifier(C=3e-3, n_jobs=num_threads, n_iter=5,
                                      loss='hinge', warm_start=True,
                                      class_weight=class_weight)
    olr = SGDClassifier(loss="log", learning_rate="optimal", penalty="l2", average=True,
                        n_iter=4, n_jobs=num_threads, class_weight=class_weight)
    llr = SGDClassifier(loss="log", learning_rate="optimal", penalty="l2", average=True,
                        n_iter=4, n_jobs=num_threads, class_weight=class_weight)
    otdt = DecisionTreeClassifier(criterion='gini', max_features=None,
                                max_depth=1, class_weight=class_weight)
    opdt = DecisionTreeClassifier(criterion='gini', max_features=None,
                                 max_depth=1, class_weight=class_weight)
    tdt = AdhocDecisionTree(class_weight[0], troll_first=True)
    pdt = AdhocDecisionTree(class_weight[0], troll_first=False)
    perceptron = SGDClassifier(loss="perceptron", eta0=1, class_weight=class_weight,
                               learning_rate="constant", penalty=None, average=True, n_iter=4)
    if args.balanced:
        pref += '_bal'

    start = (int(time.time()-(2015-1970)*365.25*24*60*60))//60
    feats = list(range(7)) + list(range(17, 33))
    fres = [[] for _ in range(20)]
    active = [{'sampling': lambda d: int(.03*d)},
              {'sampling': lambda d: int(.15*d)},
              {'sampling': lambda d: int(.28*d)},
              {'sampling': lambda d: int(.40*d)},
              {'sampling': lambda d: int(.52*d)},
              {'sampling': lambda d: int(.65*d)}]
    n, m = graph.order, len(graph.E)
    logc = 1 if n*log(n) < m else 0.4
    if logc == 1 and args.balanced:
        logc = .55
    logn = logc*log(graph.order)
    vals = [2, 4, 6, 8, logn] if pref.startswith('WIK') else [1, 2, 3, 4, logn]
    if args.balanced:
        vals = [1, 2, 3, 4, logn]
    batch = [{'batch': v} for v in vals]
    if args.balanced:
        batch = [{'batch': v/{'WIK': 1, 'SLA': 2, 'EPI': 3}[args.data]}
                  for v in vals]
    if args.online:
        fres = [online_exp(graph, pref, start, part, args.online)
                for _ in range(num_rep)]
        p.save_var('{}_online_{}_{}.my'.format(pref, start, part+1), np.array(fres))
        import sys
        sys.exit()
    cst_troll = lambdas_troll[pref]
    cst_pleas = lambdas_pleas[pref]
    for params in active if args.active else batch:
        only_troll_fixed, only_troll_learned, only_troll_transfer = [], [], []
        only_pleas_fixed, only_pleas_learned, only_pleas_transfer = [], [], []
        first_troll_learned, first_troll_transfer, first_troll_fixed = [], [], []
        first_pleas_learned, first_pleas_transfer, first_pleas_fixed = [], [], []
        ppton = []
        logreg, pa = [], []
        lesko, chiang, asym = [], [], []
        allones, randompred = [], []
        for _ in range(num_rep):
            graph.select_train_set(**params)
            Xl, yl, train_set, test_set = graph.compute_features()
            idx2edge = {i: e for e, i in graph.edge_order.items()}
            Xa, ya = np.array(Xl), np.array(yl)
            train_feat = np.ix_(train_set, feats)
            test_feat = np.ix_(test_set, feats)
            gold = ya[test_set]
            pp = (test_set, idx2edge)

            cst = [.5, 1, .0]
            pred_function = graph.train(lambda features: tree_prediction(features, cst, True))
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold, pp)
            only_troll_fixed.append(res)
            pred_function = graph.train(otdt, Xa[train_set, 15:16], ya[train_set])
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:16], gold, pp)
            only_troll_learned.append(res)
            cst = cst_troll.copy(); cst[1] = 1; cst[2] = .0
            pred_function = graph.train(lambda features: tree_prediction(features, cst, True))
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold, pp)
            only_troll_transfer.append(res)
            cst = [.5, .5, .5]
            pred_function = graph.train(lambda features: tree_prediction(features, cst, True))
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold, pp)
            first_troll_fixed.append(res)

            cst = [.5, 1, .0]
            pred_function = graph.train(lambda features: tree_prediction(features, cst, False))
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold, pp)
            only_pleas_fixed.append(res)
            pred_function = graph.train(opdt, Xa[train_set, 16:17], ya[train_set])
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 16:17], gold, pp)
            only_pleas_learned.append(res)
            cst = cst_pleas.copy(); cst[1] = 1; cst[2] = .0
            pred_function = graph.train(lambda features: tree_prediction(features, cst, False))
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold, pp)
            only_pleas_transfer.append(res)
            cst = [.5, .5, .5]
            pred_function = graph.train(lambda features: tree_prediction(features, cst, False))
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold, pp)
            first_pleas_fixed.append(res)

            pred_function = graph.train(pdt, Xa[train_set, 15:17], ya[train_set])
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold, pp)
            first_pleas_learned.append(res)
            cst = cst_pleas.copy()
            pred_function = graph.train(lambda features: tree_prediction(features, cst, False))
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold, pp)
            first_pleas_transfer.append(res)

            pred_function = graph.train(tdt, Xa[train_set, 15:17], ya[train_set])
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold, pp)
            first_troll_learned.append(res)
            cst = cst_troll.copy()
            pred_function = graph.train(lambda features: tree_prediction(features, cst, True))
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold, pp)
            first_troll_transfer.append(res)

            pred_function = graph.train(olr, Xa[train_set, 15:17], ya[train_set])
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold, pp)
            logreg.append(res)
            pred_function = graph.train(pac, Xa[train_set, 15:17], ya[train_set])
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold, pp)
            pa.append(res)

            pred_function = graph.train(llr, Xa[train_feat], ya[train_set])
            res = graph.test_and_evaluate(pred_function, Xa[test_feat], gold)
            lesko.append(res)

            pred_function = graph.train(lambda features: [0]+[1,]*(features.shape[0]-1))
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold)
            allones.append(res)
            pred_function = graph.train(lambda features: np.random.rand(features.shape[0])>.5)
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold)
            randompred.append(res)

            pred_function = graph.train(perceptron, Xa[train_set, 15:17], ya[train_set])
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold, pp)
            ppton.append(res)

            # asym.append([.8, .9, .5, .3, 2, res[-1]])
            # chiang.append([.8, .9, .5, .3, 2, res[-1]])
            # continue
            esigns = {(u, v): graph.E.get((u,v)) if (u,v) in graph.E else graph.E.get((v,u))
                      for u, adj in graph.Gfull.items() for v in adj}
            mapping={i: i for i in range(graph.order)}
            sstart = lp.clock()
            sadj, test_edges = sp.get_training_matrix(666, mapping, slcc=set(range(graph.order)),
                                                      tree_edges=graph.Esign.keys(), G=graph.Gfull,
                                                      EDGE_SIGN=esigns)
            ngold, pred = sp.predict_edges(sadj, 15, mapping, test_edges,
                                          graph.Gfull, esigns, bk=9000)
            time_elapsed = lp.clock() - sstart
            C = lp.confusion_matrix(ngold, pred)
            fp, tn = C[0, 1], C[0, 0]
            acc, fpr, f1, mcc = [lp.accuracy_score(ngold, pred),  fp/(fp+tn),
                                 lp.f1_score(ngold, pred, average='weighted', pos_label=None),
                                 lp.matthews_corrcoef(ngold, pred)]
            frac = 1 - len(test_edges)/len(graph.E)
            asym.append([acc, f1, mcc, fpr, time_elapsed, frac])

            ngold, pred, time_elapsed, frac = getWH.run_chiang(graph)
            C = lp.confusion_matrix(ngold, pred)
            fp, tn = C[0, 1], C[0, 0]
            acc, fpr, f1, mcc = [lp.accuracy_score(ngold, pred),  fp/(fp+tn),
                                 lp.f1_score(ngold, pred, average='weighted', pos_label=None),
                                 lp.matthews_corrcoef(ngold, pred)]
            chiang.append([acc, f1, mcc, fpr, time_elapsed, frac])

        fres[0].append(only_troll_fixed)
        fres[1].append(only_troll_learned)
        fres[2].append(only_troll_transfer)
        fres[3].append(only_pleas_fixed)
        fres[4].append(only_pleas_learned)
        fres[5].append(only_pleas_transfer)
        fres[6].append(first_troll_learned)
        fres[7].append(first_troll_transfer)
        fres[8].append(first_pleas_learned)
        fres[9].append(first_pleas_transfer)
        fres[10].append(logreg)
        fres[11].append(pa)
        fres[12].append(lesko)
        fres[13].append(chiang)
        fres[14].append(asym)
        fres[15].append(allones)
        fres[16].append(randompred)
        fres[17].append(ppton)
        fres[18].append(first_troll_fixed)
        fres[19].append(first_pleas_fixed)
    if args.active:
        pref += '_active'
    p.save_var('{}_{}_{}.my'.format(pref, start, part+1), (None, fres))
