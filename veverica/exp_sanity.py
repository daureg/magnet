#! /usr/bin/env python
import numpy as np
import LillePrediction as llp
from L1Classifier import L1Classifier
from math import ceil, log
from exp_tworules import pred_with_threshold, find_threshold


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
    parser.add_argument("-b", "--balanced", action='store_true',
                        help="Should there be 50/50 +/- edges")
    parser.add_argument("-a", "--active", action='store_true',
                        help="Use active sampling strategy")
    parser.add_argument("-n", "--nrep", help="number of repetition", type=int,
                        default=3)
    args = parser.parse_args()
    pref = args.data
    num_rep = args.nrep
    balanced = args.balanced

    graph = llp.LillePrediction(use_triads=False)
    graph.load_data(pref, balanced)
    dicho = L1Classifier()
    class_weight = {0: 1.4, 1: 1}
    if balanced:
        pref += '_bal'
    if args.active:
        pref += '_active'

    start = (int(time.time()-(2015-1970)*365.25*24*60*60))//60

    batch = [{'batch': v/100} for v in np.linspace(5, 100, 11).astype(int)]
    active = [{'sampling': lambda d: max(1, min(d, int(4*ceil(log(d+1))))),
               'replacement': True, 'do_out': True, 'do_in': True},
              {'sampling': lambda d: max(1, min(d, int(4*ceil(log(d+1))))),
               'replacement': False, 'do_out': True, 'do_in': True},
              {'sampling': lambda d: max(1, min(d, int(2*ceil(log(d+1))))),
               'replacement': False, 'do_out': True, 'do_in': True}]
    fres = [[] for _ in range(4)]
    res_file = '{}_{}_{}'.format(pref, start, part+1)
    params_file = '_params_' + res_file
    for params in active if args.active else batch:
        both_fixed, both_learned = [], []
        l1_learned, l1_fixed = [], []
        for _ in range(num_rep):
            graph.select_train_set(**params)
            Xl, yl, train_set, test_set = graph.compute_features()
            X, ya = np.array(Xl), np.array(yl)
            if args.active:
                np_train = np.zeros(X.shape[0], dtype=bool)
                np_train[train_set] = 1
                np_test = np.zeros(X.shape[0], dtype=bool)
                np_test[test_set] = 1
                train_set = np.logical_and(np_train, graph.in_lcc)
                test_set = np.logical_and(np_test, graph.in_lcc)
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

            pred_function = graph.train(lambda features: features[:, 0] + features[:, 1] < 1)
            res = graph.test_and_evaluate(pred_function, X[test_set, 15:17], gold, pp)
            l1_fixed.append(res)

            denom_troll = X[:, 12] + X[:, 5]
            valid_denom = denom_troll > 0
            tmp_train = np.zeros_like(valid_denom, dtype=bool)
            tmp_train[train_set] = True
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

        fres[0].append(both_fixed)
        fres[1].append(both_learned)
        fres[2].append(l1_fixed)
        fres[3].append(l1_learned)
    np.savez_compressed(res_file, res=np.array(fres))
