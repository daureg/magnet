#! /usr/bin/env python
# vim: set fileencoding=utf-8
from LillePrediction import *
from L1Classifier import L1Classifier

if __name__ == '__main__':
    # pylint: disable=C0103
    from math import log, ceil
    import time
    import socket
    import argparse
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
    parser.add_argument("-n", "--nrep", help="number of repetition", type=int,
                        default=8)
    args = parser.parse_args()
    pref = args.data
    num_rep = args.nrep
    balanced = args.balanced

    graph = LillePrediction(use_triads=False)
    graph.load_data(data[pref], args.balanced)
    dicho = L1Classifier()
    if args.balanced:
        pref += '_bal'

    start = (int(time.time()-(2015-1970)*365.25*24*60*60))//60

    params = [{'sampling': lambda d: min(1, d), 'replacement': True, 'do_out': True, 'do_in': False},
              {'sampling': lambda d: min(1, d), 'replacement': True, 'do_out': False, 'do_in': True},
              {'sampling': lambda d: max(1, min(d, int(4*ceil(log(d))))), 'replacement': True, 'do_out': True, 'do_in': False},
              {'sampling': lambda d: max(1, min(d, int(4*ceil(log(d))))), 'replacement': True, 'do_out': False, 'do_in': True},
              {'sampling': lambda d: min(1, d), 'replacement': True, 'do_out': True, 'do_in': True},
              {'sampling': lambda d: max(1, min(d, int(4*ceil(log(d))))), 'replacement': True, 'do_out': True, 'do_in': True},
              ]
    res = np.zeros((6, 6, num_rep))
    for i, p in enumerate(params):
        for r in range(num_rep):
            graph.select_train_set(**p)
            Xl, yl, train_set, test_set = graph.compute_features()
            idx2edge = {i: e for e, i in graph.edge_order.items()}
            Xa, ya = np.array(Xl), np.array(yl)
            np_train = np.zeros(Xa.shape[0], dtype=bool)
            np_train[train_set] = 1
            np_test = np.zeros(Xa.shape[0], dtype=bool)
            np_test[test_set] = 1
            train_set = np.logical_and(np_train, graph.in_lcc)
            test_set = np.logical_and(np_test, graph.in_lcc)
            gold = ya[test_set]
            pp = (test_set, idx2edge)

            if i <= 3:
                pred_function = graph.train(lambda features: features[:, i % 2] < 0.5)
                res[i, :, r] = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold, pp)
            else:
                pred_function = graph.train(dicho, Xa[train_set, 15:17], ya[train_set])
                res[i, :, r] = graph.test_and_evaluate(pred_function, Xa[test_set, 15:17], gold, pp)
    pref += '_active'
    np.savez_compressed('{}_{}_{}'.format(pref, start, part+1), res=res)
