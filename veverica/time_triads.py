import LillePrediction as llp
import numpy as np

if __name__ == '__main__':
    # pylint: disable=C0103
    import time
    import socket
    import argparse
    part = int(socket.gethostname()[-1])-1
    num_threads = 16

    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Which data to use",
                        choices={'wik', 'sla', 'epi', 'kiw', 'aut'}, default='wik')
    parser.add_argument("-n", "--nrep", help="number of repetition", type=int,
                        default=4)
    args = parser.parse_args()
    pref = args.data
    num_rep = args.nrep

    graph = llp.LillePrediction(use_triads=True)
    graph.load_data(pref, False)
    class_weight = {0: 1.4, 1: 1}
    llr = llp.SGDClassifier(loss="log", learning_rate="optimal", penalty="l2", average=True,
                            n_iter=4, n_jobs=num_threads, class_weight=class_weight)
    start = (int(time.time()-(2015-1970)*365.25*24*60*60))//60
    triads_feats = list(range(7)) + list(range(17, 33))

    batch = [{'batch': v} for v in [.15]]
    fres = [[] for _ in range(1)]
    for r, params in enumerate(batch):
        lesko, chiang, asym = [], [], []
        for _ in range(num_rep):
            graph.select_train_set(**params)
            Xl, yl, train_set, test_set = graph.compute_features()
            idx2edge = {i: e for e, i in graph.edge_order.items()}
            Xa, ya = np.array(Xl), np.array(yl)
            train_feat = np.ix_(train_set, triads_feats)
            test_feat = np.ix_(test_set, triads_feats)
            gold = ya[test_set]
            revealed = ya[train_set]
            pp = (test_set, idx2edge)

            pred_function = graph.train(llr, Xa[train_feat], ya[train_set])
            res = graph.test_and_evaluate(pred_function, Xa[test_feat], gold)
            res.append(graph.triad_time)
            res.append(graph.feature_time)
            lesko.append(res)

        fres[0].append(lesko)
    res_file = '{}_{}_{}_time'.format(pref, start, part+1)
    np.savez_compressed(res_file, res=np.array(fres))
