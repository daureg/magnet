from lprop_matrix import _train_second
from scipy.optimize import minimize
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
                        default=3)
    args = parser.parse_args()
    pref = args.data
    num_rep = args.nrep
    start = (int(time.time()-(2015-1970)*365.25*24*60*60))//60
    res_file = '{}_{}_{}'.format(pref, start, part+1)

    diameters = {'aut': 22, 'wik': 16, 'sla': 32, 'epi': 38, 'kiw': 30}
    DIAMETER = diameters[pref]
    G, E = pg.load_directed_signed_graph('directed_{}.pack'.format(pref))
    n, m = len(G), len(E)
    sorted_edges = np.zeros((m, 3), dtype=np.int)
    for i, (e, s) in enumerate(sorted(E.items())):
        u, v = e
        sorted_edges[i, :] = (u, v, s)
    ya = sorted_edges[:, 2]
    eps = 40

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

    def log_likelihood(p, q):
        t=np.maximum(p[ppf]+q[qpf], 1e-15)
        t[npf] = np.maximum(2 - t[npf], 1e-15)
        return np.log(t/2).sum()

    def solve_for_pq(x0):
        sstart = clock()
        res = minimize(cost_and_grad, x0, jac=True, bounds=bounds,
                       options=dict(maxiter=1000))
        print(clock() - sstart)
        x = res.x
        p, q = x[:n], x[n:n*2]
        feats = p[ppf]+q[qpf]
        pred = feats[test_set] > -find_threshold(-feats[train_set], ya[train_set])
        cost = res.fun
        mcc = matthews_corrcoef(gold, pred)
        return mcc, cost, log_likelihood(p, q)
    batch_p = [.03, .09, .20]
    nb_methods = 4
    num_x0 = 8
    res = np.zeros((len(batch_p), nb_methods, num_rep, len(('mcc', 'cost', 'log_likelihood'))))
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
            f, tt = _train_second(Wone, done, train_set, magnified, (m, n), False)
            feats = f[:m]
            sstart = clock()
            k = - find_threshold(-feats[train_set], revealed, True)
            tt += clock() - sstart
            pred = feats[test_set] > k
            mcc = matthews_corrcoef(gold, pred)
            pl, ql, yl = (f[m:m+n]+1)/2, (f[m+n:]+1)/2, f[:m]
            cost = cost_and_grad(np.hstack((pl, ql, yl)))[0]
            res[nb_batch, 0, nrep, :] = (mcc, cost, log_likelihood(pl, ql))

            p0 = np.random.uniform(.2,.8, n)
            p0[dout > 0] = dout_p[dout > 0] / dout[dout > 0]
            q0 = np.random.uniform(.2, .8, n)
            q0[din > 0] = din_p[din > 0]/din[din > 0]
            y0 = np.random.uniform(.2, .8, m)
            y0[train_set] = ya[train_set]
            x0 = np.hstack((p0, q0, y0))
            res[nb_batch, 1, nrep, :] = solve_for_pq(x0)

            x0 = np.hstack((pl, ql, yl)) + np.random.normal(0, .04, x0.size)
            res[nb_batch, 2, nrep, :] = solve_for_pq(x0)

            random_res = [solve_for_pq(np.random.uniform(0, 1, x0.size)) for _ in range(num_x0)]
            res[nb_batch, 3, nrep, :] = np.mean(random_res, 0)
            np.savez_compressed(res_file, res=res)
