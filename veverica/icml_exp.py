#! /usr/bin/env python
# vim: set fileencoding=utf-8
from LillePrediction import *
from exp_tworules import pred_with_threshold, find_threshold
from L1Classifier import L1Classifier
from rank_nodes import NodesRanker
from bayes_feature import compute_bayes_features
import treestar
RBFS, RTST = None, None

if __name__ == '__main__':
    # pylint: disable=C0103
    from math import log, ceil
    import time
    import socket
    import argparse
    part = int(socket.gethostname()[-1])-1
    num_threads = 16

    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Which data to use",
                        choices={'wik', 'sla', 'epi', 'kiw'}, default='wik')
    parser.add_argument("-b", "--balanced", action='store_true',
                        help="Should there be 50/50 +/- edges")
    parser.add_argument("-a", "--active", action='store_true',
                        help="Use active sampling strategy")
    parser.add_argument("-n", "--nrep", help="number of repetition", type=int,
                        default=4)
    args = parser.parse_args()
    pref = args.data
    num_rep = args.nrep
    balanced = args.balanced

    graph = LillePrediction(use_triads=True)
    graph.load_data(pref, args.balanced)
    class_weight = {0: 1.4, 1: 1}
    olr = SGDClassifier(loss="log", learning_rate="optimal", penalty="l2", average=True,
                        n_iter=4, n_jobs=num_threads, class_weight=class_weight)
    llr = SGDClassifier(loss="log", learning_rate="optimal", penalty="l2", average=True,
                        n_iter=4, n_jobs=num_threads, class_weight=class_weight)
    rnlr = SGDClassifier(loss='log', n_iter=4, class_weight=class_weight,
                         warm_start=True, average=True)
    bflr = SGDClassifier(loss='log', n_iter=4, class_weight=class_weight,
                         warm_start=True, average=True)
    perceptron = SGDClassifier(loss="perceptron", eta0=1, class_weight=class_weight,
                               learning_rate="constant", penalty=None, average=True, n_iter=4)
    dicho = L1Classifier()
    nrk = NodesRanker(autotune_budget=0)
    if args.balanced:
        pref += '_bal'

    start = (int(time.time()-(2015-1970)*365.25*24*60*60))//60
    feats = list(range(7)) + list(range(17, 33))

    cs = [{'sampling': lambda d: int(ceil(1e-7*log(d)))},
          {'sampling': lambda d: int(ceil(.5*log(d)))},
          {'sampling': lambda d: int(ceil(1*log(d)))},
          {'sampling': lambda d: int(ceil(2*log(d)))},
          {'sampling': lambda d: int(ceil(3*log(d)))},
          {'sampling': lambda d: int(ceil(4*log(d)))},
          {'sampling': lambda d: int(ceil(5*log(d)))}]

    batch = [{'batch': v/100} for v in range(15, 91, 15)]
    fres = [[] for _ in range(14)]
    for r, params in enumerate(cs if args.active else batch):
        only_troll_fixed, l1_fixed, l1_learned = [], [], []
        logreg, ppton = [], []
        lesko, chiang, asym = [], [], []
        treek, bfsl = [], []
        both_fixed, both_learned = [], []
        rank_nodes, bayes_feat = [], []
        for _ in range(num_rep):
            graph.select_train_set(**params)
            Xl, yl, train_set, test_set = graph.compute_features()
            idx2edge = {i: e for e, i in graph.edge_order.items()}
            Xa, ya = np.array(Xl), np.array(yl)
            if args.active:
                np_train = np.zeros(Xa.shape[0], dtype=bool)
                np_train[train_set] = 1
                np_test = np.zeros(Xa.shape[0], dtype=bool)
                np_test[test_set] = 1
                train_set = np.logical_and(np_train, graph.in_lcc)
                test_set = np.logical_and(np_test, graph.in_lcc)
            train_feat = np.ix_(train_set, feats)
            test_feat = np.ix_(test_set, feats)
            gold = ya[test_set]
            pp = (test_set, idx2edge)
            pp = None

            pred_function = graph.train(lambda features: features[:, 0] < 0.5)
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold, pp)
            only_troll_fixed.append(res)
            pred_function = graph.train(lambda features: features[:, 0] + features[:, 1] < 1)
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold, pp)
            l1_fixed.append(res)
            pred_function = graph.train(dicho, Xa[train_set, 15:17], ya[train_set])
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold, pp)
            l1_learned.append(res)
            frac = len(train_set)/len(graph.E)

            denom_troll = Xa[:, 12] + Xa[:, 5]
            valid_denom = denom_troll > 0
            tmp_train = np.zeros_like(valid_denom, dtype=bool)
            tmp_train[train_set] = True
            valid_train_denom = np.logical_and(valid_denom, tmp_train)
            denom_both = denom_troll + Xa[:, 9] + Xa[:, 4]
            valid_both = denom_both > 0
            valid_train_both = np.logical_and(valid_both, tmp_train)
            both_feats = (Xa[:, 12] + Xa[:, 4]) / denom_both

            pred_function = graph.train(lambda features:
                                        pred_with_threshold(features, 0.5, denom_both[test_set]==0))
            res = graph.test_and_evaluate(pred_function, both_feats[test_set], gold, pp)
            both_fixed.append(res)
            kboth = find_threshold(both_feats[valid_train_both], ya[valid_train_both])
            pred_function = graph.train(lambda features:
                                        pred_with_threshold(features, kboth, denom_both[test_set]==0))
            res = graph.test_and_evaluate(pred_function, both_feats[test_set], gold, pp)
            both_learned.append(res)

            pred_function = graph.train(olr, Xa[train_set, 15:17], ya[train_set])
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold, pp)
            logreg.append(res)
            pred_function = graph.train(perceptron, Xa[train_set, 15:17], ya[train_set])
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold, pp)
            ppton.append(res)

            if r == 0 and args.active:
                bfsl.append(treestar.baseline_bfs(graph.Gfull, graph.E))
                k = {'wik': 7, 'sla': 8, 'epi': 9,
                     'wik_bal': 9, 'sla_bal': 10, 'epi_bal': 1}[pref]
                treek.append(treestar.full_treestar(graph.Gfull, graph.E, k))

            if args.active:
                asym.append([.8, .9, .5, .3, 2, frac])
                chiang.append([.8, .9, .5, .3, 2, frac])
                lesko.append([.8, .9, .5, .3, 2, frac])
                continue

            bfsl.append([.8, .9, .5, .3, 2, frac])
            treek.append([.8, .9, .5, .3, 2, frac])

            sstart = lp.clock()
            Etrain = graph.Esign
            Etest = {e: s for e, s in graph.E.items() if e not in Etrain}
            nrk.fit(Etrain, graph.order)
            Xtrain, ytrain = nrk.transform(Etrain)
            Xtest, ytest = nrk.transform(Etest)
            rnlr.fit(Xtrain, ytrain)
            pred = rnlr.predict(Xtest)
            end = lp.clock() - sstart
            gold = ytest
            C = lp.confusion_matrix(gold, pred)
            fp, tn = C[0, 1], C[0, 0]
            acc, fpr, f1, mcc = [lp.accuracy_score(gold, pred), fp/(fp+tn),
                                 lp.f1_score(gold, pred, average='weighted', pos_label=None),
                                 lp.matthews_corrcoef(gold, pred)]
            rank_nodes.append([acc, f1, mcc, fpr, end, frac])

            sstart = lp.clock()
            Xbayes = compute_bayes_features(Xa, ya, train_set, test_set, graph)
            bflr.fit(Xbayes[train_set, :], ya[train_set])
            pred = bflr.predict(Xbayes[test_set, :])
            end = lp.clock() - sstart
            gold=ya[test_set]
            C = lp.confusion_matrix(gold, pred)
            fp, tn = C[0, 1], C[0, 0]
            acc, fpr, f1, mcc = [lp.accuracy_score(gold, pred), fp/(fp+tn),
                                 lp.f1_score(gold, pred, average='weighted', pos_label=None),
                                 lp.matthews_corrcoef(gold, pred)]
            bayes_feat.append([acc, f1, mcc, fpr, end, frac])

            pred_function = graph.train(llr, Xa[train_feat], ya[train_set])
            res = graph.test_and_evaluate(pred_function, Xa[test_feat], gold)
            lesko.append(res)
            esigns = {(u, v): graph.E.get((u, v)) if (u, v) in graph.E else graph.E.get((v, u))
                      for u, adj in graph.Gfull.items() for v in adj}
            mapping = {i: i for i in range(graph.order)}
            sstart = lp.clock()
            sadj, test_edges = sp.get_training_matrix(666, mapping, slcc=set(range(graph.order)),
                                                      tree_edges=graph.Esign.keys(), G=graph.Gfull,
                                                      EDGE_SIGN=esigns)
            ngold, pred = sp.predict_edges(sadj, 15, mapping, test_edges, graph.Gfull, esigns)
            time_elapsed = lp.clock() - sstart
            C = lp.confusion_matrix(ngold, pred)
            fp, tn = C[0, 1], C[0, 0]
            acc, fpr, f1, mcc = [lp.accuracy_score(ngold, pred), fp/(fp+tn),
                                 lp.f1_score(ngold, pred, average='weighted', pos_label=None),
                                 lp.matthews_corrcoef(ngold, pred)]
            frac = 1 - len(test_edges)/len(graph.E)
            asym.append([acc, f1, mcc, fpr, time_elapsed, frac])

            ngold, pred, time_elapsed, frac = getWH.run_chiang(graph)
            C = lp.confusion_matrix(ngold, pred)
            fp, tn = C[0, 1], C[0, 0]
            acc, fpr, f1, mcc = [lp.accuracy_score(ngold, pred), fp/(fp+tn),
                                 lp.f1_score(ngold, pred, average='weighted', pos_label=None),
                                 lp.matthews_corrcoef(ngold, pred)]
            chiang.append([acc, f1, mcc, fpr, time_elapsed, frac])

        fres[0].append(only_troll_fixed)
        fres[1].append(l1_fixed)
        fres[2].append(l1_learned)
        fres[3].append(logreg)
        fres[4].append(ppton)
        fres[5].append(lesko)
        fres[6].append(chiang)
        fres[7].append(asym)
        if r == 0:
            fres[8].append(treek)
            fres[9].append(bfsl)
        else:
            fres[8].append(list(fres[8][0]))
            fres[9].append(list(fres[9][0]))
        fres[10].append(both_fixed)
        fres[11].append(both_learned)
        fres[12].append(rank_nodes)
        fres[13].append(bayes_feat)
    if args.active:
        pref += '_active'
    res_file = '{}_{}_{}'.format(pref, start, part+1)
    np.savez_compressed(res_file, res=np.array(fres))
