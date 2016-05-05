"""Minimize an energy function over the whole graph."""
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
from timeit import default_timer as clock
from collections import defaultdict
import LillePrediction as llp
import numpy as np
import scipy.sparse as sp


def setup_problem(graph, training_edges, idx2edge):
    n, m = graph.order, len(graph.E)

    p0, q0 = np.random.uniform(.3, 1, n), np.random.uniform(.3, 1, n)
    y0 = np.random.random(m)
    fixed = np.array(sorted([graph.edge_order[e] for e in training_edges]),
                     dtype=np.uint)
    fixed_val = np.array([int(training_edges[idx2edge[i]]) for i in fixed])
    y0[fixed] = fixed_val

    data, row, col = [], [], []
    for u, neighbors in graph.Gout.items():
        signs = [1 if graph.E[(u, v)] else -1 for v in neighbors]
        row.extend((u for _ in signs))
        col.extend(neighbors)
        data.extend(signs)
    A = sp.coo_matrix((data, (row, col)), shape=(n, n), dtype=np.int8)
    aA = np.abs(A.tocsc())

    ppf, qpf = [], []
    tmp = defaultdict(list)
    dout_p = defaultdict(int)
    dout_m = defaultdict(int)
    din_p = defaultdict(int)
    din_m = defaultdict(int)
    douth = defaultdict(int)
    dinh = defaultdict(int)
    for i, ((u, v), s) in enumerate(sorted(graph.E.items())):
        ppf.append(u)
        qpf.append(v)
        tmp[v].append(i)
        if (u, v) in training_edges:
            (dout_p if s else dout_m)[u] += 1
            (din_p if s else din_m)[v] += 1
            douth[u] += 1
            dinh[u] += 1
    ppf = np.array(ppf, dtype=int)
    qpf = np.array(qpf, dtype=int)
    dout = np.array([graph.dout[u] for u in range(n)], dtype=np.uint)
    din = np.array([graph.din[u] for u in range(n)], dtype=np.uint)
    dout_p = np.array([dout_p[u] for u in range(n)], dtype=np.uint)
    dout_m = np.array([dout_m[u] for u in range(n)], dtype=np.uint)
    din_p = np.array([din_p[u] for u in range(n)], dtype=np.uint)
    din_m = np.array([din_m[u] for u in range(n)], dtype=np.uint)
    douth = np.array([douth[u] for u in range(n)], dtype=np.uint)
    dinh = np.array([dinh[u] for u in range(n)], dtype=np.uint)
    pdp = dout > 0
    pdq = din > 0
    cdp = dout.cumsum()-1
    cdq = din.cumsum()-1
    ytidx = np.array([v for i in range(max(tmp)+1) for v in tmp[i]])
    _pdp = douth > 0
    _pdq = dinh > 0
    p0[_pdp] = dout_p[_pdp]/douth[_pdp]
    q0[_pdq] = din_p[_pdq]/dinh[_pdq]
    x0 = np.hstack((p0, q0, y0))

    def cost(x):
        p, q, y = x[:n], x[n:2*n], x[2*n:]
        Aq = aA.dot(q)
        pA = aA.T.dot(p)
        pdout = p*dout
        qdin = q*din
        pppf = p[ppf]
        qqpf = q[qpf]
        c = ((p*pdout + q*qdin).sum() +
             (4*y*y + -4*pppf*y + -4*qqpf*y).sum() +
             2*p.dot(Aq))
        Y = y.cumsum()
        Yt = y[ytidx].cumsum()
        grad_p = 2*p*dout + 2*Aq.T
        grad_p[pdp] -= 4*np.diff(np.hstack(([0], Y[cdp[pdp]])))
        grad_q = 2*q*din + 2*pA
        grad_q[pdq] -= 4*np.diff(np.hstack(([0], Yt[cdq[pdq]])))
        grad_y = 8*y -4*(pppf + qqpf)
        grad_y[fixed] = 0
        return c, np.hstack((grad_p, grad_q, grad_y))

    bounds = [(0, 1) for _ in x0]
    return cost, x0, bounds

if __name__ == "__main__":
    import sys
    start = clock()
    pref = sys.argv[1]
    print('Loading graph {}…'.format(pref))
    graph = llp.LillePrediction(use_triads=False)
    graph.load_data(pref)
    n = graph.order
    idx2edge = {i: e for e, i in graph.edge_order.items()}
    print('…in {:.3f}s'.format(clock() - start)); start = clock()

    batch = 0.5
    print('Sampling {}% edges…'.format(100*batch))
    es = graph.select_train_set(batch=batch)
    nes = set(graph.E.keys())-set(es.keys())
    test_edges = np.array(sorted([graph.edge_order[e] for e in nes]))
    test_val = np.array([int(graph.E[idx2edge[i]]) for i in test_edges])
    print('…in {:.3f}s'.format(clock() - start)); start = clock()

    print('Setting up the problem…')
    cost_function, sol_init, bounds = setup_problem(graph, es, idx2edge)
    print('…in {:.3f}s'.format(clock() - start)); start = clock()

    print('Solving the problem…')
    res = minimize(cost_function, sol_init, jac=True, bounds=bounds,
                   options=dict(maxiter=2000))
    print('…in {:.3f}s'.format(clock() - start)); start = clock()
    print(res)

    x = res.x
    ps, qs, ys = x[:n], x[n:2*n], x[2*n:]
    k = 0.5
    pred = (ys[test_edges] > k).astype(int)
    gold = test_val
    C = confusion_matrix(gold, pred)
    fp, tn = C[0, 1], C[0, 0]
    print([accuracy_score(gold, pred),
           f1_score(gold, pred, average='weighted', pos_label=None),
           matthews_corrcoef(gold, pred), fp/(fp+tn)])
