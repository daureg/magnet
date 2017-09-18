# vim: set fileencoding=utf-8
"""Test MCC w.r.t p & q on synthetic generated sign labeling."""
import random
import time

import numpy as np

import lprop_matrix as lm
from exp_tworules import find_threshold
from L1Classifier import L1Classifier
from LillePrediction import LillePrediction


def setup_label_propagation(graph, name):
    diameters = {'aut': 22, 'wik': 16, 'wik_ts': 16, 'sla': 32,
                 'epi': 38, 'epi_ts': 38, 'kiw': 30, 'adv': 10}
    lm.DIAMETER = diameters[name]
    P, sorted_edges = lm.compute_gprime_mat(graph.Gfull, graph.E)
    W, d, sorted_edges = lm.compute_gsecond_mat(graph.Gfull, graph.E)
    m = sorted_edges.shape[0]
    n = (P.shape[0] - m) // 2
    return m, n, W, d, sorted_edges


tau = 3
A = 1/(1-np.exp(-tau))


def non_linear_mapping(x):
    # return (-1*x*x/3 + 4/3*x) # 7/12~.5833 at .5
    # return (-2 * x * x / 3 + 5 / 3 * x)  # 2/3~.6666 at .5
    # return min(1, (-5*x*x/4 + 9/4*x)) #  y=.7 at x=.4
    return A*(1-np.exp(-tau*x))


def generate_params(nodes, degrees, sizes, easy):
    n, m = sizes
    z = np.random.permutation(len(nodes))
    param = np.zeros(n)
    intervals = [0, ] + list(np.cumsum(degrees[z]) / m)
    for (a, b), u in zip(zip(intervals, intervals[1:]), z):
        if easy:
            param[nodes[u]] = int(b <= 0.75)
        else:
            param[nodes[u]] = np.random.uniform(non_linear_mapping(a), non_linear_mapping(b))
    return param


def label_the_graph(graph, easy=False):
    sizes = (graph.order, len(graph.E))
    Vout = sorted(graph.Gout)
    dout = np.array([graph.dout[u] for u in Vout])
    p = generate_params(Vout, dout, sizes, easy)
    Vin = sorted(graph.Gin)
    din = np.array([graph.din[u] for u in Vin])
    q = generate_params(Vin, din, sizes, easy)
    for u, v in graph.E:
        graph.E[(u, v)] = int(random.random() < .5 * (p[u] + q[v]))
    return p, q


if __name__ == '__main__':
    # pylint: disable=C0103
    import argparse
    import socket
    part = int(socket.gethostname()[-1]) - 1
    num_threads = 15
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Which data to use", default='wik',
                        choices={'wik', 'wik_ts', 'sla', 'epi', 'epi_ts', 'kiw', 'aut', 'adv'})
    parser.add_argument("-s", "--size", type=float, default=.1, help="Fraction of training size")
    parser.add_argument("-n", "--nrep", help="number of repetition", type=int, default=3)
    parser.add_argument("-g", "--ngen", help="number of generation", type=int, default=10)
    parser.add_argument("-e", "--easy", action='store_true', help="Use easier distribution")
    args = parser.parse_args()
    name = args.data
    easy = args.easy
    num_rep = args.nrep
    num_gen = args.ngen
    frac = args.size / 100 if args.size > 1 else args.size

    dicho = L1Classifier()

    start = (int(time.time() - (2017 - 1970) * 365.25 * 24 * 60 * 60)) // 60
    res_file = '{}_{}_{}_{}_{}'.format(name, int(100 * frac),
                                       'easy' if easy else 'uni', start, part + 1)
    graph = LillePrediction(use_triads=False)
    graph.load_data(name)
    m, n, W, d, sorted_edges = setup_label_propagation(graph, name)
    lres, bres = None, None
    batch_level = [.2, .4, .8]
    gres = np.zeros((len(batch_level)*num_gen*num_rep, 4))
    row = 0
    for frac in batch_level:
        for gen in range(num_gen):
            P, Q = label_the_graph(graph, easy)
            m, n, W, d, sorted_edges = setup_label_propagation(graph, name)
            for _ in range(num_rep):
                graph.select_train_set(batch=frac)
                Xl, yl, train_set, test_set = graph.compute_features()
                Xa, ya = np.array(Xl), np.array(yl)
                gold = ya[test_set]
                revealed = ya[train_set]
                us, vs = sorted_edges[test_set, 0], sorted_edges[test_set, 1]

                f, _ = lm._train_second(W, d, train_set, 2 * revealed - 1, (m, n))
                feats = f[:m]
                k_star = -find_threshold(-feats[train_set], ya[train_set], True)
                pred = feats[test_set] > k_star
                tres = np.vstack((P[us], Q[vs], f[m:m+n][us], f[m+n:][vs],
                                  Xa[test_set, 33], Xa[test_set, 34],
                                  gold, pred)).T
                lres = tres if lres is None else np.vstack((lres, tres))

                dicho.fit(Xa[train_set, 15:17], ya[train_set])
                pred = dicho.predict(Xa[test_set, 15:17])
                tres = np.vstack((P[us], Q[vs], Xa[test_set, 15], Xa[test_set, 16],
                                  Xa[test_set, 33], Xa[test_set, 34],
                                  gold, pred)).T
                bres = tres if bres is None else np.vstack((bres, tres))
                gres[row, :] = (len(test_set), np.bincount(revealed)[1]/revealed.size,
                                dicho.k, k_star)
                row += 1
                np.savez_compressed(res_file, lres=lres, bres=bres, gres=gres)
