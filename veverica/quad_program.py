# vim: set fileencoding=utf-8
"""."""
from collections import defaultdict
from itertools import combinations, product
from operator import itemgetter
from scipy.spatial.distance import pdist, squareform
import cvxpy as cvx
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import prettyplotlib as ppl

USERS, ITEMS, RATINGS = None, None, None
DIM_SIZE = 6
FEATURE_START = 4


def load_data():
    global USERS, ITEMS, RATINGS
    data = np.load('sphere/final_movielens.npz')
    USERS, ITEMS, RATINGS = itemgetter('users', 'movies', 'ratings')(data)
    ITEMS = ITEMS[:2929, :]


def build_graph():
    """Link similar nodes through the items on which they agree"""
    global USERS, ITEMS, RATINGS
    csim = pdist(RATINGS, 'cosine')
    fdist = squareform(csim)
    np.fill_diagonal(fdist, 2)
    i = 0
    res = []
    nodes = set()
    for u, v in zip(*np.unravel_index(fdist.argsort(None)[:1800],
                                      fdist.shape)):
        i += 1
        if (i % 2) == 0:
            continue
        common = np.argwhere(np.logical_and(np.logical_and(RATINGS[u, :] > 0,
                                                           RATINGS[v, :] > 0),
                                            RATINGS[u, :] == RATINGS[v, :]))
        if len(common) < 15:
            continue
        user_diff = USERS[u, :] - USERS[v, :]
        nodes.add(u)
        nodes.add(v)
        # Take the 5 with smallest dot value
        itms_idx = list(common.ravel())
        dot_value = [np.dot(user_diff, ITEMS[_, :])**2 for _ in itms_idx]
        for common_items_idx in np.argsort(dot_value)[:5]:
            res.append((u, v, itms_idx[common_items_idx],
                        dot_value[common_items_idx]))
    return sorted(res), nodes


def build_random_graph(upper_size):
    """Link users at random on random items"""
    rres = []
    n, m = USERS.shape[0], ITEMS.shape[0]
    users_pairs = list(combinations(range(n), 2))
    items_ids = list(range(m))
    while len(rres) < upper_size:
        u, v = random.choice(users_pairs)
        for common_items in random.sample(items_ids, random.randint(2, 6)):
            rres.append((u, v, common_items,
                         np.dot(USERS[u, :] - USERS[v, :],
                                ITEMS[common_items, :])**2))
    return rres


def plot_users_similarity(log_scale=True, *edges):
    sim = [_[3] for _ in edges[0]]
    rsim = [_[3] for _ in edges[1]]
    if log_scale:
        plt.hist(np.log10(sim), 50, label='cosine', alpha=.5)
        plt.hist(np.log10(rsim), 50, label='random', alpha=.5)
        plt.xlabel('users similarity along edge x_ij (log10 scale)')
    else:
        plt.hist(sim, 50, label='cosine', alpha=.5)
        plt.hist(rsim, 50, label='random', alpha=.5)
        plt.xlabel('users similarity along edge x_ij')
    plt.ylabel('count')
    ppl.legend()


def hide_some_users(nodes, fraction_of_hidden=.35):
    hidden, shown, visibility = set(), set(), {}
    for u in nodes:
        if random.random() < fraction_of_hidden:
            hidden.add(u)
            visibility[u] = False
        else:
            shown.add(u)
            visibility[u] = True
    user_order = sorted(hidden)
    return user_order, visibility


def construct_optimization_matrix(edges, visibility, user_order):
    global USERS, ITEMS, DIM_SIZE
    d = DIM_SIZE
    nn = len(user_order)*d
    Q = np.zeros((nn, nn))
    c = np.zeros((nn, 1))
    for i, j, x, _ in edges:
        if i > j:
            i, j = j, i
        oi, oj = i, j
        ui = USERS[oi, FEATURE_START:]
        uj = USERS[oj, FEATURE_START:]
        xij = ITEMS[x, FEATURE_START:]
        i_known = visibility[i]
        j_known = visibility[j]
        i_unknown = not i_known
        j_unknown = not j_known
        if i_known and j_known:
            continue
        if i_unknown:
            i = user_order.index(i)
        if j_unknown:
            j = user_order.index(j)
        if i_unknown and j_unknown:
            # both vectors are unknown
            # add cross product terms
            for k, l in product(range(d), range(d)):
                Q[d*i+k, d*j+l] += -xij[k]*xij[l]
        if i_unknown:
            # add ui.xij ² contribution
            for k, l in combinations(range(d), 2):
                Q[d*i+k, d*i+l] += xij[k]*xij[l]
            for k in range(d):
                Q[d*i+k, d*i+k] += xij[k]*xij[k]/2
        if j_unknown:
            # add uj⋅xij ² contribution
            for k, l in combinations(range(d), 2):
                Q[d*j+k, d*j+l] += xij[k]*xij[l]
            for k in range(d):
                Q[d*j+k, d*j+k] += xij[k]*xij[k]/2
        if i_known:
            # add cross term when u_i is constant
            ui = USERS[oi, FEATURE_START:]
            ui_xij = np.dot(ui, xij)
            c[d*j:d*(j+1)] += np.reshape(-2*xij.T*ui_xij, (d, 1))
        if j_known:
            # add cross term when u_j is constant
            uj = USERS[oj, FEATURE_START:]
            uj_xij = np.dot(uj, xij)
            c[d*i:d*(i+1)] += np.reshape(-2*xij.T*uj_xij, (d, 1))

    # symmetrize Q and add some regularization on ||u|| (i.e make sure
    # eigenvalues are positive
    Qs = Q.T + Q
    regul = np.linalg.eigvalsh(Qs).min()
    if regul < 0:
        regul *= -1.03
    else:
        regul = 0
    print(regul)
    return Q.T + Q + 0.1*np.eye(nn), c


def solve_problem(Q, c, verbose=False):
    u = cvx.Variable(c.size)
    obj = cvx.Minimize(cvx.quad_form(u, Q)+cvx.sum_entries(c.T*u))
    prob = cvx.Problem(obj)
    prob.solve(verbose=verbose, solver=cvx.SCS)
    uval = np.array(u.value).ravel()
    return np.reshape(uval, (c.size // DIM_SIZE, DIM_SIZE))


def analyze_solution(recovered_users, hidden_user_idx, edges, verbose=False,
                     drawing=False):
    global USERS
    adj = defaultdict(set)
    for i, j, _, _ in edges:
        adj[i].add(j)
        adj[j].add(i)
    recovered_users /= np.sqrt((recovered_users ** 2).sum(-1))[..., np.newaxis]
    gold_users = USERS[hidden_user_idx, FEATURE_START:]
    gold_users /= np.sqrt((gold_users ** 2).sum(-1))[..., np.newaxis]
    diff = np.sqrt(((gold_users - recovered_users) ** 2).sum(-1))
    non_zeros = np.where(recovered_users[:, 0] > -100)[0]
    if verbose:
        print('average distance {:.3f}'.format(np.mean(diff[non_zeros])))
        prct = [5, 25, 50, 75, 95]
        vals = np.percentile(diff[non_zeros], prct)
        print('Percentile: '+'\t'.join(['{}'.format(str(_).ljust(5))
                                        for _ in prct]))
        print('            '+'\t'.join(['{:.3f}'.format(_) for _ in vals]))

    embeddings = np.zeros((4, non_zeros.size))
    i = 0
    for uidx in range(len(recovered_users)):
        neighbors = adj[hidden_user_idx[uidx]]
        hidden_neighbors = {_ for _ in neighbors if _ in hidden_user_idx}
        tot_dst = 0
        me = USERS[uidx, FEATURE_START:]
        me /= np.linalg.norm(me)
        for n in neighbors:
            nei = USERS[n, FEATURE_START:]
            tot_dst += np.linalg.norm(nei/np.linalg.norm(nei) - me)
        if uidx in non_zeros:
            embeddings[:, i] = [diff[uidx], len(neighbors),
                                len(hidden_neighbors)/len(neighbors),
                                tot_dst/len(neighbors)]
            i += 1

    if drawing:
        labels = ['number of neighbors',
                  'fraction of unknown neighbors',
                  'mean distance from all neighbors']
        for i in range(1, 4):
            with sns.plotting_context("notebook", font_scale=1.7,
                                      rc={"figure.figsize": (20, 10)}):
                sns.regplot(embeddings[i, :], embeddings[0, :],
                            label=labels[i-1])
        ppl.legend()
    return embeddings


def construct_1d_optimization_matrix(edges, visibility, user_order,
                                     projected_users, projected_items):
    nn = len(user_order)
    Q = np.zeros((nn, nn))
    c = np.zeros((nn, 1))
    for i, j, x, _ in edges:
        if i > j:
            i, j = j, i
        oi, oj = i, j
        ui = projected_users[oi]
        uj = projected_users[oj]
        xij = projected_items[x]**2
        i_known = visibility[i]
        j_known = visibility[j]
        i_unknown = not i_known
        j_unknown = not j_known
        if i_known and j_known:
            continue
        if i_unknown:
            i = user_order.index(i)
        if j_unknown:
            j = user_order.index(j)
        if i_unknown and j_unknown:
            # both vectors are unknown
            # add cross product terms
            Q[i, j] = -xij
        if i_unknown:
            # add ui.xij ² contribution
            Q[i, i] = xij/2
        if j_unknown:
            # add uj⋅xij ² contribution
            Q[j, j] = xij/2
        if i_known:
            # add cross term when u_i is constant
            c[j] = -2*ui*xij
        if j_known:
            # add cross term when u_j is constant
            c[i] = -2*uj*xij

    # symmetrize Q and add some regularization on ||u|| (i.e make sure
    # eigenvalues are positive
    Qs = Q.T + Q
    regul = np.linalg.eigvalsh(Qs).min()
    if regul < 0:
        regul *= -1.03
    else:
        regul = 0
    print(regul)
    return Q.T + Q + regul*np.eye(nn), c


def solve_on_random_basis(edges, visibility, hidden_user_idx, users, items):
    import scipy
    global DIM_SIZE
    old_dim = DIM_SIZE
    DIM_SIZE = 1
    is_a_basis = False
    d = users.shape[1]
    while not is_a_basis:
        A = 2*(np.random.random((d, d)) > .5).astype(int)-1
        A[:, 0] = 1
        is_a_basis = abs(np.linalg.det(A)) > 0
    projected_items = np.dot(A, items.T)
    projected_users = np.dot(A, users.T)
    print(projected_users.shape)
    B = np.zeros((d, len(hidden_user_idx)))
    for i in range(d):
        Q, c = construct_1d_optimization_matrix(edges, visibility,
                                                hidden_user_idx,
                                                projected_users[i, :],
                                                projected_items[i, :])
        # TODO is the regularization changing for each dimension a problem?
        B[i, :] = solve_problem(Q, c).ravel()
    DIM_SIZE = old_dim
    return scipy.linalg.solve(A, B).T
