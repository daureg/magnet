#! /usr/bin/env python
# vim: set fileencoding=utf-8
import scipy.sparse as ssp
from scipy.optimize import minimize

import lprop_matrix as lm
import treestar
from bayes_feature import compute_bayes_features
from exp_tworules import find_threshold, pred_with_threshold
from L1Classifier import L1Classifier
from online_sign_pred import prepare_data, online_maxnorm_completion, run_icml_exp
from LillePrediction import *
from rank_nodes import NodesRanker
from exp_params import diameters, batch

RBFS, RTST = None, None

if __name__ == '__main__':
    # pylint: disable=C0103
    from math import log, ceil
    import time
    import socket
    import argparse
    part = int(socket.gethostname()[-1])-1
    num_threads = 15
    do_asym = False

    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Which data to use", default='wik', choices=set(diameters))
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
    dicho_mcc = L1Classifier(maximize_mcc=True)
    nrk = NodesRanker(autotune_budget=0)

    lm.DIAMETER = diameters[pref]
    data = lm.sio.loadmat('{}_gprime.mat'.format(pref))
    P, sorted_edges = data['P'], data['sorted_edges']
    data = lm.sio.loadmat('{}_gsecond.mat'.format(pref), squeeze_me=True)
    W, d = data['W'], data['d']
    ya = sorted_edges[:, 2]
    m = sorted_edges.shape[0]
    n = (P.shape[0] - m)//2

    # variables needed for the cost & grad function
    dout, din = (np.zeros(n, dtype=np.uint), np.zeros(n, dtype=np.uint))
    A_row, A_col, A_data = [], [], []
    Bp_row, Bp_col, Bp_data = [], [], []
    Bq_row, Bq_col, Bq_data = [], [], []
    ppf, qpf = [], []
    for ve, row in enumerate(sorted_edges):
        i, u, v, s = ve, *row
        dout[u] += 1
        din[v] += 1
        ppf.append(u)
        qpf.append(v)
        A_row.append(u)
        A_col.append(v)
        A_data.append(1)
        Bp_row.append(u)
        Bp_col.append(i)
        Bp_data.append(1)
        Bq_row.append(v)
        Bq_col.append(i)
        Bq_data.append(1)
    ppf = np.array(ppf, dtype=int)
    qpf = np.array(qpf, dtype=int)
    aA = ssp.coo_matrix((A_data, (A_row, A_col)), shape=(n, n), dtype=np.int).tocsc()
    Bp = ssp.coo_matrix((Bp_data, (Bp_row, Bp_col)), shape=(n, m), dtype=np.int).tocsc()
    Bq = ssp.coo_matrix((Bq_data, (Bq_row, Bq_col)), shape=(n, m), dtype=np.int).tocsc()
    bounds = [(0.0, 1.0) for _ in range(m+2*n)]

    def solve_for_pq(x0):
        sstart = lp.clock()
        res = minimize(cost_and_grad, x0, jac=True, bounds=bounds,
                       options=dict(maxiter=1500))
        y = res.x[2*n:]
        time_elapsed = lp.clock() - sstart
        return y, time_elapsed

    if args.balanced:
        pref += '_bal'

    start = (int(time.time()-(2015-1970)*365.25*24*60*60))//60
    triads_feats = list(range(7)) + list(range(17, 33))

    cs = [{'sampling': lambda d: int(ceil(1e-7*log(d)))},
          {'sampling': lambda d: int(ceil(.5*log(d)))},
          {'sampling': lambda d: int(ceil(1*log(d)))},
          {'sampling': lambda d: int(ceil(2*log(d)))},
          {'sampling': lambda d: int(ceil(3*log(d)))},
          {'sampling': lambda d: int(ceil(4*log(d)))},
          {'sampling': lambda d: int(ceil(5*log(d)))}]

    fres = [[] for _ in range(37)]
    for r, params in enumerate(cs if args.active else batch):
        only_troll_fixed, l1_fixed, l1_learned = [], [], []
        only_troll_fixed_raw, l1_fixed_raw, l1_learned_raw = [], [], []
        logreg, ppton = [], []
        logreg_raw, ppton_raw = [], []
        both_fixed, both_learned = [], []
        both_fixed_raw, both_learned_raw = [], []
        lpmin_erm, lpmin_neg = [], []
        lpmin_erm_raw, lpmin_neg_raw = [], []
        lesko, chiang, asym = [], [], []
        rank_nodes, bayes_feat = [], []
        treek, bfsl = [], []
        l1_learned_mcc, both_learned_mcc, lpmin_erm_mcc = [], [], []
        l1_learned_mcc_raw, both_learned_mcc_raw, lpmin_erm_mcc_raw = [], [], []
        lpmin_second, lpmin_second_raw = [], []
        quad_optim, quad_optim_raw = [], []
        maxnorm, maxnorm_raw = [], []
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
            train_feat = np.ix_(train_set, triads_feats)
            test_feat = np.ix_(test_set, triads_feats)
            gold = ya[test_set]
            revealed = ya[train_set]
            pp = (test_set, idx2edge)

            def cost_and_grad(x):
                p, q, y = x[:n], x[n:2*n], x[2*n:]
                Aq = aA.dot(q)
                pA = aA.T.dot(p)
                c = ((p*p*dout + q*q*din).sum() + 2*p.dot(Aq))
                t = p[ppf] + q[qpf]
                c += (4*y*(y-t)).sum()
                grad_p = 2*p*dout + 2*Aq.T - 4*Bp@y
                grad_q = 2*q*din + 2*pA - 4*Bq@y
                grad_y = 4*(2*y - t)
                grad_y[train_set] = 0
                return c, np.hstack((grad_p, grad_q, grad_y))

            Asym, edges_matrix_indices, strain, stest, mngold = prepare_data(graph)
            sstart = lp.clock()
            U, V, _, _ = online_maxnorm_completion(Asym, edges_matrix_indices, graph.order)
            feats_test, k_star = run_icml_exp(U, V, strain, stest)
            time_elapsed = lp.clock() - sstart
            pred_function = graph.train(lambda features: features > k_star)
            graph.time_used = time_elapsed
            res = graph.test_and_evaluate(pred_function, feats_test, mngold, pp)
            maxnorm.append(res)
            graph.time_used = time_elapsed
            maxnorm_raw.append(graph.test_and_evaluate(pred_function, feats_test, mngold))

            x0 = np.random.uniform(0, 1, m+2*n)
            x0[2*n:][train_set] = ya[train_set]
            feats, time_elapsed = solve_for_pq(x0)
            sstart = lp.clock()
            sorted_y = np.sort(feats[test_set])
            pos_frac = 1-ya[train_set].sum()/len(train_set)
            k_star = sorted_y[int(pos_frac*sorted_y.size)]
            time_elapsed += lp.clock() - sstart
            pred_function = graph.train(lambda features: features > k_star)
            graph.time_used = time_elapsed
            res = graph.test_and_evaluate(pred_function, feats[test_set], gold, pp)
            quad_optim.append(res)
            graph.time_used = time_elapsed
            quad_optim_raw.append(graph.test_and_evaluate(pred_function, feats[test_set], gold))

            pred_function = graph.train(lambda features: features[:, 0] < 0.5)
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold, pp)
            graph.time_used = 0
            only_troll_fixed_raw.append(graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold))
            only_troll_fixed.append(res)
            pred_function = graph.train(lambda features: features[:, 0] + features[:, 1] < 1)
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold, pp)
            graph.time_used = 0
            l1_fixed_raw.append(graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold))
            l1_fixed.append(res)

            pred_function = graph.train(dicho, Xa[train_set, 15:17], ya[train_set])
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold, pp)
            graph.time_used = 0
            l1_learned_raw.append(graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold))
            l1_learned.append(res)
            pred_function = graph.train(dicho_mcc, Xa[train_set, 15:17], ya[train_set])
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold, pp)
            graph.time_used = 0
            l1_learned_mcc_raw.append(graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold))
            l1_learned_mcc.append(res)
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
            graph.time_used = 0
            both_fixed_raw.append(graph.test_and_evaluate(pred_function, both_feats[test_set], gold))
            both_fixed.append(res)
            #TODO not counting time for finding k here
            kboth = find_threshold(both_feats[valid_train_both], ya[valid_train_both])
            pred_function = graph.train(lambda features:
                                        pred_with_threshold(features, kboth, denom_both[test_set]==0))
            res = graph.test_and_evaluate(pred_function, both_feats[test_set], gold, pp)
            graph.time_used = 0
            both_learned_raw.append(graph.test_and_evaluate(pred_function, both_feats[test_set], gold))
            both_learned.append(res)

            kboth_mcc = find_threshold(both_feats[valid_train_both],
                                       ya[valid_train_both], True)
            pred_function = graph.train(lambda features:
                                        pred_with_threshold(features, kboth_mcc, denom_both[test_set]==0))
            res = graph.test_and_evaluate(pred_function, both_feats[test_set], gold, pp)
            graph.time_used = 0
            both_learned_mcc_raw.append(graph.test_and_evaluate(pred_function, both_feats[test_set], gold))
            both_learned_mcc.append(res)

            pred_function = graph.train(olr, Xa[train_set, 15:17], ya[train_set])
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold, pp)
            graph.time_used = 0
            logreg_raw.append(graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold))
            logreg.append(res)
            pred_function = graph.train(perceptron, Xa[train_set, 15:17], ya[train_set], False)
            res = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold, pp)
            graph.time_used = 0
            ppton_raw.append(graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold))
            ppton.append(res)

            feats, time_elapsed = lm._train(P, sorted_edges, train_set, ya[train_set], (m, n))
            sstart = lp.clock()
            k_star = -find_threshold(-feats[train_set], ya[train_set])
            time_taken = time_elapsed + lp.clock() - sstart
            pred_function = graph.train(lambda features: features>k_star)
            graph.time_used = time_taken
            res = graph.test_and_evaluate(pred_function, feats[test_set], gold, pp)
            lpmin_erm.append(res)
            graph.time_used = time_taken
            lpmin_erm_raw.append(graph.test_and_evaluate(pred_function, feats[test_set], gold))

            sstart = lp.clock()
            k_star_mcc = -find_threshold(-feats[train_set], ya[train_set], True)
            time_taken = time_elapsed + lp.clock() - sstart
            pred_function = graph.train(lambda features: features>k_star_mcc)
            graph.time_used = time_taken
            res = graph.test_and_evaluate(pred_function, feats[test_set], gold, pp)
            lpmin_erm_mcc.append(res)
            graph.time_used = time_taken
            lpmin_erm_mcc_raw.append(graph.test_and_evaluate(pred_function, feats[test_set], gold))

            negative_frac = np.bincount(revealed)[0]/revealed.size
            scores = np.sort(feats[test_set])
            k_neg = scores[int(negative_frac*scores.size)]
            graph.time_used = time_elapsed
            pred_function = graph.train(lambda features: features>k_neg)
            res = graph.test_and_evaluate(pred_function, feats[test_set], gold, pp)
            lpmin_neg.append(res)
            graph.time_used = time_elapsed
            lpmin_neg_raw.append(graph.test_and_evaluate(pred_function, feats[test_set], gold))

            f, time_elapsed = lm._train_second(W, d, train_set, 2*revealed-1, (m, n))
            feats = f[:m]
            sstart = lp.clock()
            k_star = -find_threshold(-feats[train_set], ya[train_set], True)
            time_elapsed += lp.clock() - sstart
            pred_function = graph.train(lambda features: features > k_star)
            graph.time_used = time_elapsed
            res = graph.test_and_evaluate(pred_function, feats[test_set], gold, pp)
            lpmin_second.append(res)
            graph.time_used = time_elapsed
            lpmin_second_raw.append(graph.test_and_evaluate(pred_function, feats[test_set], gold))

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

            Etrain = graph.Esign
            Etest = {e: s for e, s in graph.E.items() if e not in Etrain}
            nrk.fit(Etrain, graph.order)
            Xtrain, ytrain = nrk.transform(Etrain)
            Xtest, ytest = nrk.transform(Etest)
            sstart = lp.clock()
            rnlr.fit(Xtrain, ytrain)
            feats_train = rnlr.predict_proba(Xtrain)[:, 1]
            k = -find_threshold(-feats_train, ytrain, True)
            feats_test = rnlr.predict_proba(Xtest)[:, 1]
            pred = feats_test > k
            end = lp.clock() - sstart + nrk.time_taken
            gold = ytest
            C = lp.confusion_matrix(gold, pred)
            fp, tn = C[0, 1], C[0, 0]
            acc, fpr, f1, mcc = [lp.accuracy_score(gold, pred), fp/(fp+tn),
                                 lp.f1_score(gold, pred, average='weighted', pos_label=None),
                                 lp.matthews_corrcoef(gold, pred)]
            rank_nodes.append([acc, f1, mcc, fpr, end, frac])

            Xbayes, time_taken = compute_bayes_features(Xa, ya, train_set, test_set, graph)
            sstart = lp.clock()
            bflr.fit(Xbayes[train_set, :], ya[train_set])
            feats_train = bflr.predict_proba(Xbayes[train_set, :])[:, 1]
            k = -find_threshold(-feats_train, ya[train_set], True)
            feats_test = bflr.predict_proba(Xbayes[test_set, :])[:, 1]
            pred = feats_test > k
            end = lp.clock() - sstart + time_taken
            gold = ya[test_set]
            C = lp.confusion_matrix(gold, pred)
            fp, tn = C[0, 1], C[0, 0]
            acc, fpr, f1, mcc = [lp.accuracy_score(gold, pred), fp/(fp+tn),
                                 lp.f1_score(gold, pred, average='weighted', pos_label=None),
                                 lp.matthews_corrcoef(gold, pred)]
            bayes_feat.append([acc, f1, mcc, fpr, end, frac])

            pred_function = graph.train(llr, Xa[train_feat], ya[train_set], False)
            # graph.time_used += graph.triad_time
            res = graph.test_and_evaluate(pred_function, Xa[test_feat], gold)
            lesko.append(res)

            if do_asym:
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
            else:
                asym.append([.8, .9, .5, .3, 2, frac])

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
        fres[14].append(lpmin_erm)
        fres[15].append(lpmin_neg)
        fres[16].append(only_troll_fixed_raw)
        fres[17].append(l1_fixed_raw)
        fres[18].append(l1_learned_raw)
        fres[19].append(logreg_raw)
        fres[20].append(ppton_raw)
        fres[21].append(both_fixed_raw)
        fres[22].append(both_learned_raw)
        fres[23].append(lpmin_erm_raw)
        fres[24].append(lpmin_neg_raw)
        fres[25].append(l1_learned_mcc)
        fres[26].append(l1_learned_mcc_raw)
        fres[27].append(both_learned_mcc)
        fres[28].append(both_learned_mcc_raw)
        fres[29].append(lpmin_erm_mcc)
        fres[30].append(lpmin_erm_mcc_raw)
        fres[31].append(lpmin_second)
        fres[32].append(lpmin_second_raw)
        fres[33].append(quad_optim)
        fres[34].append(quad_optim_raw)
        fres[35].append(maxnorm)
        fres[36].append(maxnorm_raw)
    if args.active:
        pref += '_active'
    res_file = '{}_{}_{}'.format(pref, start, part+1)
    np.savez_compressed(res_file, res=np.array(fres))
