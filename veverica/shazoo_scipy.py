import os

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from future.utils import iteritems
from scipy.sparse.linalg import bicgstab


def get_weight_matrix(E, dataset='citeseer'):
    if os.path.exists('{}.mat'.format(dataset)):
        data = sio.loadmat('{}.mat'.format(dataset), squeeze_me=True)
        return data['W'], data['d']
    m = len(E)
    sorted_edges = np.zeros((m, 3))
    for i, (e, s) in enumerate(sorted(iteritems(E))):
        u, v = e
        sorted_edges[i, :] = (u, v, s)
    n = sorted_edges[:, :2].astype(int).max() + 1
    W_row, W_col, W_data = [], [], []
    for ve, row in enumerate(sorted_edges):
        u, v, s = int(row[0]), int(row[1]), row[2]
        W_row.extend((u, v))
        W_col.extend((v, u))
        W_data.extend((s, s))
    W = sp.coo_matrix((W_data, (W_row, W_col)), shape=(n, n),
                      dtype=np.float64).tocsc()
    d = np.array(np.abs(W).sum(1)).ravel()
    d[d == 0] = 1
    d = 1/d
    sio.savemat('{}.mat'.format(dataset), dict(W=W, d=d), do_compression=True)
    return W, d


def run_labprop(gold_signs, sorted_test_set, sorted_train_set, W, d):
    fixed_vals = np.array([gold_signs[u] for u in sorted_train_set], dtype=float)
    f = np.zeros(len(gold_signs))
    f[sorted_train_set] = fixed_vals
    for iter_ in range(20):
        f = (W@f)*d
        f[sorted_train_set] = fixed_vals
    return np.sign(f[sorted_test_set])


def solve_by_zeroing_derivative(mapped_E, mapped_El_L, mapping, L, reorder=True):
    n = len(mapping)
    if n == 0:
        # no edges mean I have only one node
        assert len(L) == 1
        return L, 0
    n_internal = n-len(L)
    W_data, W_row, W_col = [], [], []
    b = np.zeros(n)
    for u, v, w in mapped_E:
        W_row.extend((u, u, v, v))
        W_col.extend((u, v, u, v))
        W_data.extend((2*w, -2*w, -2*w, 2*w))
    for u, l, w in mapped_El_L:
        u = int(u)
        W_row.append(u)
        W_col.append(u)
        W_data.append(2*w)
        b[u] += 2*w*l
    W = sp.coo_matrix((W_data, (W_row, W_col)), shape=(n, n)).tocsc()
    if reorder:
        r = sp.csgraph.reverse_cuthill_mckee(W, symmetric_mode=True)
        nmapping = {v: i for i, v in enumerate(r)}
        mWrow = [nmapping[_] for _ in W_row]
        mWcol = [nmapping[_] for _ in W_col]
        W = sp.coo_matrix((W_data, (mWrow, mWcol)), shape=(n, n),).tocsc()
        x = bicgstab(W[:n_internal, :n_internal], b[r][:n_internal])
        xx = np.zeros(n)
        for pos_in_x, real_idx in enumerate(r):
            if pos_in_x < n_internal:
                xx[real_idx] = x[0][pos_in_x]
    else:
        xx = bicgstab(W, b)[0]
    res = {}
    for u, pos_in_x in mapping.items():
        if u not in L:
            res[u] = xx[pos_in_x]
    return res, np_cost_l2(xx, mapped_E, mapped_El_L)


def np_cost_l2(x, mapped_E, mapped_El_L):
    if mapped_E.size > 0:
        internal = np.sum(mapped_E[:, 2]*((x[mapped_E[:, 0].astype(int)] - x[mapped_E[:, 1].astype(int)])**2))
    else:
        internal = 0
    border = np.sum(mapped_El_L[:, 2]*((x[mapped_El_L[:, 0].astype(int)] - mapped_El_L[:, 1])**2))
    return internal + border
