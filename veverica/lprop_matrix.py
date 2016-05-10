from exp_tworules import find_threshold
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
from timeit import default_timer as clock
import numpy as np
import pack_graph as pg
import random
import scipy.io as sio
import scipy.sparse as sp


def save_gprime(pref):
    G, E = pg.load_directed_signed_graph('directed_{}.pack'.format(pref))
    n, m =len(G),len(E)
    sorted_edges = np.zeros((m, 3), dtype=np.int)
    for i, (e, s) in enumerate(sorted(E.items())):
        u, v = e
        sorted_edges[i, :] = (u, v, s)
    W_row, W_col = [], []
    for ve, row in enumerate(sorted_edges):
        u, v, s = row
        vpi = u + m
        vqj = v + m + n
        W_row.extend((vpi, ve, ve, vqj))
        W_col.extend((ve, vpi, vqj, ve))
    W_data = np.ones_like(W_row)
    W = sp.coo_matrix((W_data, (W_row, W_col)), shape=(m+2*n, m+2*n),
                      dtype=np.uint).tocsc()
    d = np.array(W.sum(1)).ravel()
    d[d==0] = 1
    Dinv = sp.diags(1/d, format='csc')
    P = Dinv@W
    sio.savemat('{}_gprime.mat'.format(pref), dict(P=P, sorted_edges=sorted_edges),
                do_compression=True)


def _train(P, train_idx, train_y):
    f = np.random.random(P.shape[1])
    f[train_idx] = train_y
    for i in range(200):
        nf = P@f
        nf[train_idx] = train_y
        diff = np.abs(f - nf)
        if diff.sum()/nf.sum() < 5e-4:
            f = nf
            break
        f = nf
    p, q = f[m:m+n], f[m+n:]
    feats = (p[sorted_edges[:, 0]]+q[sorted_edges[:, 1]])/2
    return feats


def _inner_cv(P, train_idx, train_y, test_idx, test_y, measure):
    feats = _train(P, train_idx, train_y)
    k = -find_threshold(-feats[test_idx], test_y)
    pred = (feats[test_idx] > k)    
    return k, measure(test_y, pred)


def evaluate(feats, gold, k, time_taken, frac):
    pred = feats > k
    C = confusion_matrix(gold, pred)
    fp, tn = C[0, 1], C[0, 0]
    return (accuracy_score(gold, pred),
            f1_score(gold, pred, average='weighted', pos_label=None),
            matthews_corrcoef(gold, pred), fp/(fp+tn), time_taken, frac, k)

if __name__ == "__main__":
    import time
    import socket
    import argparse
    part = int(socket.gethostname()[-1])-1

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

    data = sio.loadmat('{}_gprime.mat'.format(pref))
    P, sorted_edges = data['P'], data['sorted_edges']
    ya = sorted_edges[:, 2]
    m = sorted_edges.shape[0]
    n = (P.shape[0] - m)//2

    if balanced:
        pref += '_bal'

    start = (int(time.time()-(2015-1970)*365.25*24*60*60))//60

    batch_p = [.025, .05, .075, .1, .15, .25, .35, .45, .55, .7, .8, .9]

    fres = [[] for _ in range(8)]
    res_file = '{}_{}_{}'.format(pref, start, part+1)

    for batch in batch_p:
        fixed_half, fixed_tt, frac_neg = [], [], []
        training_one, training_cv, testing_opt = [], [], []
        training_mcc, testing_mcc = [], []
        for _ in range(num_rep):
            train_set, test_set = [], []
            for i in range(m):
                (train_set if random.random() < batch else test_set).append(i)
            train_set = np.array(train_set)
            test_set = np.array(test_set)
            gold = ya[test_set]
            revealed = ya[train_set]
            frac = revealed.size/m

            k_half = .5
            k_two_third = 2/3

            sstart = clock()
            feats = _train(P, train_set, revealed)
            training_time = clock() - sstart

            fixed_half.append(evaluate(feats[test_set], gold, k_half, training_time, frac))
            fixed_tt.append(evaluate(feats[test_set], gold, k_two_third, training_time, frac))

            sstart = clock()
            negative_frac = np.bincount(revealed)[0]/revealed.size
            scores = np.sort(feats[test_set])
            k_neg = scores[int(negative_frac*scores.size)]
            extra = clock() - sstart
            frac_neg.append(evaluate(feats[test_set], gold, k_neg, training_time+extra, frac))

            sstart = clock()
            k_star = -find_threshold(-feats[train_set], revealed)
            extra = clock() - sstart
            training_one.append(evaluate(feats[test_set], gold, k_star, training_time+extra, frac))

            sstart = clock()
            k_star_m = -find_threshold(-feats[train_set], revealed, True)
            extra = clock() - sstart
            training_mcc.append(evaluate(feats[test_set], gold, k_star_m, training_time+extra, frac))

            sstart = clock()
            k_opt = -find_threshold(-feats[test_set], gold)
            extra = clock() - sstart
            testing_opt.append(evaluate(feats[test_set], gold, k_opt, training_time+extra, frac))

            sstart = clock()
            k_opt_m = -find_threshold(-feats[test_set], gold, True)
            extra = clock() - sstart
            testing_mcc.append(evaluate(feats[test_set], gold, k_opt_m, training_time+extra, frac))

            sstart = clock()
            cv_res = []
            for train_idx, test_idx in StratifiedKFold(revealed, n_folds=5):
                train_y, test_y = ya[train_idx], ya[test_idx]
                cv_res.append(_inner_cv(P, train_idx, train_y, test_idx, test_y, accuracy_score))
            cv_res=np.array(cv_res)
            k_cv = cv_res[np.argmax(cv_res[:, 1]), 0]
            extra = clock() - sstart
            training_cv.append(evaluate(feats[test_set], gold, k_cv, training_time+extra, frac))
        fres[0].append(fixed_half)
        fres[1].append(fixed_tt)
        fres[2].append(frac_neg)
        fres[3].append(training_one)
        fres[4].append(training_cv)
        fres[5].append(testing_opt)
        fres[6].append(training_mcc)
        fres[7].append(testing_mcc)
    # if args.active:
    #     pref += '_active'
        np.savez_compressed(res_file, res=np.array(fres))
