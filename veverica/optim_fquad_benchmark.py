import random
from timeit import default_timer as clock

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from sklearn.metrics import matthews_corrcoef

import lprop_matrix as lm
import pack_graph as pg

if __name__ == "__main__":
    from exp_tworules import find_threshold
    import time
    import socket
    import argparse
    part = int(socket.gethostname()[-1])-1

    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Which data to use",
                        choices={'wik', 'sla', 'epi', 'kiw', 'aut'})
    args = parser.parse_args()
    pref = args.data
    start = (int(time.time()-(2015-1970)*365.25*24*60*60))//60
    res_file = '{}_{}_{}'.format(pref, start, part+1)

    diameters = {'aut': 22, 'wik': 16, 'sla': 32, 'epi': 38, 'kiw': 30}
    lm.DIAMETER = diameters[pref]
    G, E = pg.load_directed_signed_graph('directed_{}.pack'.format(pref))
    n, m = len(G), len(E)
    sorted_edges = np.zeros((m, 3), dtype=np.int)
    for i, (e, s) in enumerate(sorted(E.items())):
        u, v = e
        sorted_edges[i, :] = (u, v, s)
    ya = sorted_edges[:, 2]
    eps = 2

    W_row, W_col, W_data = [], [], []
    dout, din = (np.zeros(n, dtype=np.uint), np.zeros(n, dtype=np.uint))
    dout_p, din_p = (np.zeros(n, dtype=np.uint), np.zeros(n, dtype=np.uint))
    A_row, A_col, A_data = [], [], []
    Bp_row, Bp_col, Bp_data = [], [], []
    Bq_row, Bq_col, Bq_data = [], [], []
    ppf, qpf = [], []
    for ve, row in enumerate(sorted_edges):
        i, u, v, s = ve, *row
        vpi = u + m
        vqj = v + m + n
        W_row.extend((vpi, ve,  ve,  vqj, vpi, vqj))
        W_col.extend((ve,  vpi, vqj, ve,  vqj, vpi))
        W_data.extend((eps, eps, eps, eps, -1,  -1))
        dout[u] += 1
        din[v] += 1
        if s > 0:
            dout_p[u] += 1
            din_p[v] += 1
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
    Wone = sp.coo_matrix((W_data, (W_row, W_col)), shape=(m+2*n, m+2*n),
                         dtype=np.float64).tocsc()
    done = np.array(np.abs(Wone).sum(1)).ravel()
    done[done == 0] = 1
    done = 1/done
    ppf = np.array(ppf, dtype=int)
    qpf = np.array(qpf, dtype=int)
    npf = ya < .5
    aA = sp.coo_matrix((A_data, (A_row, A_col)), shape=(n, n), dtype=np.int).tocsc()
    Bp = sp.coo_matrix((Bp_data, (Bp_row, Bp_col)), shape=(n, m), dtype=np.int).tocsc()
    Bq = sp.coo_matrix((Bq_data, (Bq_row, Bq_col)), shape=(n, m), dtype=np.int).tocsc()
    bounds = [(0.0, 1.0) for _ in range(m+2*n)]

    def solve_for_pq(x0, method='L-BFGS-B', bounds=bounds):
        sstart = clock()
        res = minimize(cost_and_grad, x0, jac=True, bounds=bounds,
                       method=method, options=dict(maxiter=1500))
        x = res.x
        p, q, y = x[:n], x[n:n*2], x[2*n:]
        feats = p[ppf]+q[qpf]
        pred = feats[test_set] > -find_threshold(-feats[train_set], ya[train_set])
        time_elapsed = clock() - sstart
        mcc = matthews_corrcoef(gold, pred)

        sorted_y = np.sort(y[test_set])
        frac = 1-ya[train_set].sum()/train_set.size
        pred_y_frac = y[test_set] > sorted_y[int(frac*sorted_y.size)]
        mcc_y_frac = matthews_corrcoef(gold, pred_y_frac)

        pred_y_fixed = y[test_set] > 0.5
        mcc_y_fixed = matthews_corrcoef(gold, pred_y_fixed)
        cost = res.fun
        return mcc, cost, mcc_y_fixed, mcc_y_frac, time_elapsed, x

    batch = 0.03
    nb_methods = 1
    num_x0 = 30
    train_set, test_set = [], []
    for i, row in enumerate(sorted_edges):
        u, v, s = row
        if random.random() < batch:
            train_set.append(i)
        else:
            test_set.append(i)
    train_set = np.array(train_set)
    test_set = np.array(test_set)
    gold = ya[test_set]
    revealed = ya[train_set]
    frac = revealed.size/m
    magnified = (2*revealed-1)

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
    f, tt = lm._train_second(Wone, done, train_set, magnified, (m, n), False)
    feats = f[:m]
    sstart = clock()
    k = - find_threshold(-feats[train_set], revealed, True)
    pred = feats[test_set] > k
    tt += clock() - sstart
    mcc = matthews_corrcoef(gold, pred)
    pl, ql, yl = (f[m:m+n]+1)/2, (f[m+n:]+1)/2, f[:m]
    cost = cost_and_grad(np.hstack((pl, ql, yl)))[0]
    res = np.array((mcc, cost, mcc, mcc, tt))

    random_res = np.zeros((2, num_x0, 5+f.size))
    for nx in range(num_x0):
        x0 = np.random.uniform(0, 1, f.size)
        x0[2*n:][train_set] = ya[train_set]
        random_res[0, nx, :] = np.hstack(solve_for_pq(x0))
        # random_res[1, nx, :] = np.hstack(solve_for_pq(x0, bounds=None))
        random_res[1, nx, :] = np.hstack(solve_for_pq(x0, method='Newton-CG', bounds=None))
    np.savez_compressed(res_file, res=res, ores=random_res,
                        train_set=train_set)
