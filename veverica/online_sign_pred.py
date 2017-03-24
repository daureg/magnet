"""Implement the method described in
Jing Wang, Jie Shen, Ping Li, and Huan Xu. 2017.
Online Matrix Completion for Signed Link Prediction. WSDM '17, pp 475-484.
https://doi.org/10.1145/3018661.3018681
"""
import matplotlib as mpl
mpl.use('Agg')
import random
from collections import defaultdict
from math import exp, sqrt

import numpy as np


def online_maxnorm_completion(observed, edges_matrix_indices, lmbda=1.2, d=50, alpha=0.5,
                              max_epochs=50, track_rmse=False):
    rmses = []
    B = defaultdict(lambda: defaultdict(int))
    n = max((u for u, v in edges_matrix_indices)) + 1
    U, V = np.random.randn(n, d), np.random.randn(n, d)
    tau, a, b = 0, 0, 0
    for nb_epoch in range(max_epochs):
        if track_rmse:
            rmse = 0
            for i, j in edges_matrix_indices:
                bij = B[i][j]
                rmse += (observed[i][j] - (1 if bij == 1 else -1))**2
            rmses.append(sqrt(rmse/len(edges_matrix_indices)))
        random.shuffle(edges_matrix_indices)
        for i, j in edges_matrix_indices:
            zij = observed[i][j]
            xij = U[i, :]@V[j, :].T
            bij = 1 if xij >= tau else 0
            B[i][j] = bij
            a += bij * zij
            b += bij + zij
            if b != 0:
                tau = a/b
            e = exp(-zij*xij)
            up = -zij*e/(1+e)
            U[i, :] -= alpha*up*V[j, :]
            norm_u = np.sqrt((U[i, :]**2).sum())
            if norm_u > lmbda:
                U[i, :] /= norm_u
            V[j, :] -= alpha*up*U[i, :]
            norm_v = np.sqrt((V[j, :]**2).sum())
            if norm_v > lmbda:
                V[j, :] /= norm_v
    return U, V, rmses


if __name__ == "__main__":
    import seaborn as sns
    import LillePrediction as lp
    from sklearn.metrics import matthews_corrcoef
    import matplotlib.pyplot as plt
    from timeit import default_timer as clock
    sns.set('talk', 'whitegrid', 'Set1', font_scale=1.5,
            rc={'figure.figsize': (24, 13), 'lines.linewidth': 5})

    graph = lp.LillePrediction(use_triads=False)
    graph.load_data('wik')
    graph.select_train_set(batch=.1)
    Asym = graph.get_partial_adjacency()
    edges_matrix_indices = sorted({(u, v) for u, adj in Asym.items() for v in adj})
    train_edges = set(edges_matrix_indices)
    test_edges = {e: s for e, s in graph.E.items() if e not in train_edges}
    stest = sorted(test_edges)
    gold = [2*int(test_edges[e])-1 for e in stest]
    stest = np.array(stest)

    armses = {}
    for alpha in np.logspace(-2, np.log10(5), 6):
        start = clock()
        U, V, rmses = online_maxnorm_completion(Asym, edges_matrix_indices, alpha=alpha,
                                                max_epochs=30, track_rmse=True)
        tmp = U.dot(V.T)
        print('{:.3f}:\t{:.3f}'.format(alpha, clock()-start))
        pred = np.sign(tmp[stest[:, 0], stest[:, 1]]).astype(int)
        armses[alpha] = (np.array(rmses), matthews_corrcoef(gold, pred))

    for al, (res, mcc) in sorted(armses.items()):
        plt.plot(res/res[0], label='$\\alpha={:.3g}$, {:.2f}'.format(al, 100*mcc))
    plt.legend()
    sns.despine()
    plt.savefig('maxnorm.pdf', frameon=True, transparent=False, bbox_inches='tight',
                pad_inches=0)
