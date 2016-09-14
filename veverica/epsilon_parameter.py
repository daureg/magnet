"""Look at the effect of the epsilon on MCC performance."""
from lprop_matrix import _train_second
from sklearn.metrics import matthews_corrcoef
from timeit import default_timer as clock
import numpy as np
import pack_graph as pg
import random
import scipy.sparse as sp

if __name__ == "__main__":
    from exp_tworules import find_threshold
    import time
    import socket
    import argparse
    part = int(socket.gethostname()[-1])-1

    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Which data to use",
                        choices={'wik', 'sla', 'epi', 'kiw', 'aut'})
    parser.add_argument("-n", "--nrep", help="number of repetition", type=int,
                        default=10)
    args = parser.parse_args()
    pref = args.data
    num_rep = args.nrep
    balanced = False

    diameters = {'aut': 22, 'wik': 16, 'sla': 32, 'epi': 38, 'kiw': 30}
    DIAMETER = diameters[pref]
    G, E = pg.load_directed_signed_graph('directed_{}.pack'.format(pref))
    n, m = len(G), len(E)
    sorted_edges = np.zeros((m, 3), dtype=np.int)
    for i, (e, s) in enumerate(sorted(E.items())):
        u, v = e
        sorted_edges[i, :] = (u, v, s)
    ya = sorted_edges[:, 2]
    epsilons = list(np.logspace(-1, 2.3, 50))
    epsilons.extend((2, 40))
    epsilons = np.sort(epsilons)
    idx_eps2 = np.where(epsilons == 2)[0][0]
    idx_eps40 = np.where(epsilons == 40)[0][0]

    start = (int(time.time()-(2015-1970)*365.25*24*60*60))//60
    batch_p = [.03, .09, .15, .20, .25]
    fres = [[] for _ in range(3)]
    res_file = '{}_{}_{}'.format(pref, start, part+1)

    res = np.zeros((len(batch_p), len(epsilons), num_rep))
    for nb_batch, batch in enumerate(batch_p):
        print(batch, pref)
        for nrep in range(num_rep):
            train_set, test_set = [], []
            for i in range(m):
                (train_set if random.random() < batch else test_set).append(i)
            train_set = np.array(train_set)
            test_set = np.array(test_set)
            gold = ya[test_set]
            revealed = ya[train_set]
            frac = revealed.size/m
            magnified = (2*revealed-1)

            for nb_eps, eps in enumerate(epsilons):
                W_row, W_col, W_data = [], [], []
                for ve, row in enumerate(sorted_edges):
                    u, v, s = row
                    vpi = u + m
                    vqj = v + m + n
                    W_row.extend((vpi, ve,  ve,  vqj, vpi, vqj))
                    W_col.extend((ve,  vpi, vqj, ve,  vqj, vpi))
                    W_data.extend((eps, eps, eps, eps, -1,  -1))
                Wone = sp.coo_matrix((W_data, (W_row, W_col)), shape=(m+2*n, m+2*n),
                                     dtype=np.float64).tocsc()
                done = np.array(np.abs(Wone).sum(1)).ravel()
                done[done == 0] = 1
                done = 1/done

                f, tt = _train_second(Wone, done, train_set, magnified, (m, n), False)
                feats = f[:m]
                sstart = clock()
                k = - find_threshold(-feats[train_set], revealed, True)
                tt += clock() - sstart
                pred = feats[test_set] > k
                res[nb_batch, nb_eps, nrep] = matthews_corrcoef(gold, pred)
            np.savez_compressed(res_file, res=res)
