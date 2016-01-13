#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Implementation of the low rank matrix completion with sigmoid loss described in
Chiang, K., Hsieh, C., Natarajan, N., Dhillon, I. S., & Tewari, A. (2014).
Prediction and Clustering in Signed Networks: A Local to Global Perspective.
Journal of Machine Learning Research, 15, 1177–1213
http://jmlr.org/papers/v15/chiang14a.html."""
import numpy as np
from math import exp

def getWH(A, rank_of_A=7, eta0=0.9, lmbda=1e-2):
    n = A.shape[0]
    edges = np.argwhere(A)
    E = edges.shape[0]
    MAX_ITER = 6*E
    W = np.random.rand(n, rank_of_A)
    H = np.random.rand(n, rank_of_A)
    rand_edges = np.random.randint(0, E-1, MAX_ITER)
    for t in range(MAX_ITER):
        i, j = edges[rand_edges[t]]
        aval = A[i, j]        
        whij = W[i,:].dot(H[j,:].T)
        chosen_loss=1/(1+exp(aval*whij))        
        save_wi = W[i, :]
        # eta = eta0 / (1 + (t//E)/(MAX_ITER//E))
        eta = eta0
        step = eta*aval*chosen_loss*(1-chosen_loss)
        W[i,:] = (1-lmbda*eta)*W[i, :] + step*H[j,:]
        H[j,:] = (1-lmbda*eta)*H[j, :] + step*save_wi
    return W, H

if __name__ == '__main__':
    # pylint: disable=C0103
    with np.load('wik_sym.npz') as f:
        Asym = f['A']
    import persistent as p
    edg = p.load_var('wik_sym_edges.my')
    from timeit import default_timer as clock
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    from sklearn.metrics import confusion_matrix
    k = 10
    s = len(edg)//k
    for i in range(k):
        test = np.array(edg[i*s:(i+1)*s])    
        gold = np.sign(Asym[test[:,0], test[:,1]])
        Asym[test[:,0], test[:,1]] = 0
        # train, pred, evaluate
        start = clock()
        W, H = getWH(Asym)
        print(clock() - start)
        tmp = W.dot(H.T)
        pred = np.sign(tmp[test[:,0], test[:,1]])
        C = confusion_matrix(gold, pred)
        fp, tn = C[0, 1], C[0, 0]
        print([accuracy_score(gold, pred), f1_score(gold, pred),
                matthews_corrcoef(gold, pred), fp/(fp+tn)])
        Asym[test[:,0], test[:,1]] = gold
