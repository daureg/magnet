#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Implementation of the low rank matrix completion with sigmoid loss described in
Chiang, K., Hsieh, C., Natarajan, N., Dhillon, I. S., & Tewari, A. (2014).
Prediction and Clustering in Signed Networks: A Local to Global Perspective.
Journal of Machine Learning Research, 15, 1177–1213
http://jmlr.org/papers/v15/chiang14a.html."""
from math import exp
from timeit import default_timer as clock

import numpy as np


def getWH(A, n, edg, rank_of_A=7, eta0=0.9, lmbda=1e-2):
    # edges = np.array(edg+[(v, u) for u, v in edg])
    edges = edg
    E = len(edges)
    MAX_ITER = 10*E
    W = np.random.rand(n, rank_of_A)
    H = np.random.rand(n, rank_of_A)
    rand_edges = np.random.randint(0, E-1, MAX_ITER)
    for t in range(MAX_ITER):
        i, j = edges[rand_edges[t]]
        aval = A[i][j]
        whij = W[i,:].dot(H[j,:].T)
        chosen_loss=1/(1+exp(aval*whij))        
        save_wi = W[i, :]
        # eta = eta0 / (1 + (t//E)/(MAX_ITER//E))
        eta = eta0
        step = eta*aval*chosen_loss*(1-chosen_loss)
        W[i,:] = (1-lmbda*eta)*W[i, :] + step*H[j,:]
        H[j,:] = (1-lmbda*eta)*H[j, :] + step*save_wi
    return W, H


def run_chiang(graph):
    Asym = graph.get_partial_adjacency()
    edges_matrix_indices = sorted({(u, v) for u, adj in Asym.items() for v in adj})
    start = clock()
    W, H = getWH(Asym, graph.order, edges_matrix_indices)
    tmp = W.dot(H.T)
    time_elapsed = clock() - start
    train_edges = set(edges_matrix_indices)
    test_edges = {e: s for e, s in graph.E.items() if e not in train_edges}
    stest = sorted(test_edges)
    gold = [2*int(test_edges[e])-1 for e in stest]
    stest = np.array(stest)
    pred = np.sign(tmp[stest[:, 0], stest[:, 1]]).astype(int)
    frac = len(train_edges)/(2*len(graph.E))
    return gold, pred, time_elapsed, frac

if __name__ == '__main__':
    # pylint: disable=C0103
    import persistent as p
    edg = p.load_var('wik_sym_edges.my')
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    from sklearn.metrics import confusion_matrix

    # import LillePrediction as llp
    # graph = llp.LillePrediction(use_triads=False)
    # graph.load_data(llp.lp.DATASETS.Wikipedia, balanced=False)
    # graph.select_train_set(sampling=lambda d: int(.3*d))
    # print(len(graph.Esign)/len(graph.E))
    # gold, pred, time_elapsed, frac = run_chiang(graph)
    # print(time_elapsed)
    # C = confusion_matrix(gold, pred)
    # fp, tn = C[0, 1], C[0, 0]
    # print([accuracy_score(gold, pred), f1_score(gold, pred, average='weighted', pos_label=None),
    #             matthews_corrcoef(gold, pred), fp/(fp+tn)])
    # import sys
    # sys.exit()

    with np.load('wik_sym.npz') as f:
        Asym = f['A']
    k = 10
    s = len(edg)//k
    for i in range(k):
        test = np.array(edg[i*s:(i+1)*s])    
        gold = np.sign(Asym[test[:,0], test[:,1]])
        Asym[test[:,0], test[:,1]] = 0
        # train, pred, evaluate
        start = clock()
        W, H = getWH(Asym, Asym.shape[0], np.argwhere(Asym))
        print(clock() - start)
        tmp = W.dot(H.T)
        pred = np.sign(tmp[test[:,0], test[:,1]])
        C = confusion_matrix(gold, pred)
        fp, tn = C[0, 1], C[0, 0]
        print([accuracy_score(gold, pred), f1_score(gold, pred, average='weighted', pos_label=None),
                matthews_corrcoef(gold, pred), fp/(fp+tn)])
        Asym[test[:,0], test[:,1]] = gold
        break
