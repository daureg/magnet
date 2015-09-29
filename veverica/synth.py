# vim: set fileencoding=utf-8
# pylint: disable=no-member
from collections import Counter, defaultdict
from grid_stretch import add_edge
from heap.heap import heap
from itertools import combinations, product
from sklearn.cluster import SpectralClustering
from sklearn.metrics import mean_squared_error, f1_score
from warnings import warn
import baseline as bl
import cvxpy as cvx
import numpy as np
import random


def create_graph(users, xs):
    # Another way of doing, which assign optimal direction to each edges
    within_size, across_size_ = 4.0, 1.0
    n = users.shape[0]
    C = np.dot(users, xs.T)
    heads, tails = zip(*combinations(range(n), 2))
    heads = np.array(heads)
    tails = np.array(tails)
    D = np.abs(C[heads,:] - C[tails,:])
    not_relevant = np.logical_or(np.abs(C[heads,:])<1e-4,
                                 np.abs(C[tails,:])<1e-4)
    D[D<1e-5] = 2
    D[not_relevant] = 2
    sc = SpectralClustering(n_clusters=4, affinity='precomputed', n_init=20)
    best_dirs = np.argsort(D, 1).astype(int)
    final_E = defaultdict(list)
    for k in range(1):
        dir_edges = create_edges_for_one_dir(xs, D, best_dirs[:, k], sc,
                                             within_size, across_size_, heads,
                                             tails)
        for (u, v), d in dir_edges.items():
            final_E[(u, v)].append(d)
    return final_E


def create_edges_for_one_dir(xs, D, best_dir, sc, within_size, across_size_,
                             heads, tails):
    final_E = {}
    for dir_id in range(xs.shape[0]):
        mask = D[best_dir == dir_id, dir_id] < 2
        t = min(1, np.percentile(D[best_dir == dir_id, dir_id][mask], 50))
        mask = np.logical_and(best_dir == dir_id, D[:, dir_id] < t)
        edges_idx = np.where(mask)[0]
        node_in_dir = set(tails[edges_idx]).union(set(heads[edges_idx]))
        this_dir_to_nodes = {i: v for i, v in enumerate(sorted(node_in_dir))}
        nodes_to_this_dir = {v: k for k, v in this_dir_to_nodes.items()}

        dn = len(node_in_dir)
        A, G, E = np.zeros((dn, dn)), {}, set()
        for i, u, v in zip(edges_idx, heads[edges_idx], tails[edges_idx]):
            u, v = nodes_to_this_dir[u], nodes_to_this_dir[v]
            if u > v:
                u, v = v, u
            add_edge(G, u, v)
            E.add((u, v))
            A[u, v] = D[i, dir_id]
            A[v, u] = D[i, dir_id]
        sc.fit(A)
        labels = sc.labels_
        cluster_sizes = Counter(labels)
        inner_target = {c: within_size*s for c, s in cluster_sizes.items()}
        across_target = {(a, b):
                         across_size_*(cluster_sizes[a]+cluster_sizes[a])/2
                         for a, b in list(combinations(cluster_sizes, 2))}
        smallest_edges = sorted(edges_idx, key=lambda i: D[i, dir_id])

        nG, nE = {}, set()
        inner_size = {c: 0 for c in inner_target}
        across_size = {c: 0 for c in across_target}
        for i, u, v in zip(smallest_edges,
                           heads[smallest_edges], tails[smallest_edges]):
            u, v = nodes_to_this_dir[u], nodes_to_this_dir[v]
            if u > v:
                u, v = v, u
            cu, cv = labels[u], labels[v]
            add_within = cu == cv and inner_size[cu] < inner_target[cu]
            pair = (cu, cv) if cu < cv else (cv, cu)
            add_across = cu != cv and across_size[pair] < across_target[pair]
            if add_within or add_across:
                add_edge(nG, u, v)
                nE.add((u, v))
                if add_within:
                    inner_size[cu] += 1
                else:
                    across_size[pair] += 1
            done_within = all((v >= inner_target[k]
                               for k, v in inner_size.items()))
            done_across = all((v >= across_target[k]
                               for k, v in across_size.items()))
            if done_within and done_across:
                break
        final_E.update({(this_dir_to_nodes[u], this_dir_to_nodes[v]): dir_id
                        for u, v in nE})

    return final_E


def to_graph_tool_simple(orig):
    """transform a adjacency dict into a graph tool structure, suitable for
    cc_pivot.draw_clustering"""
    import graph_tool as gt
    graph = gt.Graph(directed=False)
    mapping = {v: i for i, v in enumerate(G.keys())}
    graph.add_vertex(len(mapping))
    for u, adj in orig.items():
        u = mapping[u]
        for v in adj:
            v = mapping[v]
            if u < v:
                graph.add_edge(u, v)
    return graph, mapping


def generate_random_data(n=300, d=4, nx=6, sparse=False):
    """There are $n$ nodes, each associated with a $d$ dimensional unit norm
    feature vectors. In addition we pick $nx$ directions at random in
    $\mathbb{R}^{+d}$.
    If sparse is True, set between $d-10$ and $d-3$ features to 0"""
    def norm_rand_matrix(rows, columns, sparse=False):
        _ = 2*np.random.random((rows, columns))-1
        if sparse:
            for i in range(rows):
                num_zero_features = random.randint(d-10, d-3)
                zeroed = np.random.permutation(d)[:num_zero_features]
                _[i, zeroed] = 0
        return _ / np.sqrt((_ ** 2).sum(-1))[..., np.newaxis]
    assert not sparse or d > 10, "need more feature to make matrix sparse"
    users = norm_rand_matrix(n, d, sparse)
    xs = np.abs(norm_rand_matrix(nx, d))
    return users, xs


def laplacian_from_edges(E, n, nodes_to_idx=None, nb_dir=None):
    """Compute the `nb_dir` Laplacians of `E`"""
    nb_dir = nb_dir or len(set((k for dirs in E.values() for k in dirs)))
    L = np.zeros((nb_dir, n, n), dtype=int)
    for (u, v), dirs in E.items():
        if nodes_to_idx:
            u, v = nodes_to_idx[u], nodes_to_idx[v]
        for k in dirs:
            L[k][u][u] += 1
            L[k][v][v] += 1
            L[k][u][v] = -1
            L[k][v][u] = -1
    return L


def compute_loss(users, xs, L):
    nx = xs.shape[0]
    weight = np.dot(np.ones(nx), np.dot(xs, xs.T))
    lap = np.zeros(nx)
    for m, xm in enumerate(xs):
        vm = np.dot(users, xm[:, np.newaxis])
        lap[m] = np.dot(vm.T, np.dot(L[m], vm)).ravel()[0]
    return np.dot(lap, weight)


def setup_constraints(users, vis, noise_level=0.0):
    """Fix the value of visible users and return variable to optimize,
    potentially adding some noise to them"""
    n, d = users.shape
    Ux = cvx.Variable(n*d)
    constraints = []
    for u, visible in vis.items():
        if visible:
            real_user = np.copy(users[u, :])
            if noise_level > 0:
                noise = np.random.normal(scale=noise_level, size=d)
                real_user += noise
                real_user /= np.linalg.norm(real_user)
            constraints.append(Ux[u*d:(u+1)*d] == real_user)
    return Ux, constraints


def setup_optim(users, xs, L, Ux, constraints):
    """Compute all terms needed to define the optimization problem"""
    (n, d), nx = users.shape, xs.shape[0]
    weight = np.dot(np.ones(nx), np.dot(xs, xs.T))
    laps = []
    for m, xm in enumerate(xs):
        k = np.matrix(np.kron(np.eye(n), xm[:, np.newaxis]))
        vm = k.T*Ux
        laps.append(cvx.quad_form(vm, L[m]))
    obj = cvx.Minimize(sum((w*l for w, l in zip(weight, laps))))
    return cvx.Problem(obj, constraints)


def solve_optim(prob, Ux, verbose=True, normalize=True):
    """return the solution of the problem"""
    d = prob.constraints[0].size[0]
    n = Ux.size[0]//d
    loss = prob.solve(verbose=verbose, solver=cvx.SCS)
    uval = np.array(Ux.value).ravel()
    urecovered = uval.reshape(n, d)
    if normalize:
        urecovered /= np.sqrt((urecovered ** 2).sum(-1))[..., np.newaxis]
        if verbose:
            warn('you need to compute the loss yourself')
    return urecovered, loss


def evaluate_solution(users, urecovered, observed_index, xs=None, E=None,
                      hidden_edges=None):
    """Evaluate the quality of the recovered user profile"""
    mse = mean_squared_error(users[observed_index, :],
                             urecovered[observed_index, :])
    if hidden_edges is None or len(hidden_edges) < 1:
        return mse, None
    # TODO: sort directions actually
    gold = np.array([E[e] for e in sorted(hidden_edges)])
    eh = sorted(hidden_edges)
    heads, tails = zip(*eh)
    Cr = np.dot(urecovered, xs.T)
    Dr = np.abs(Cr[heads, :] - Cr[tails, :])
    pred = np.argmin(Dr, 1).astype(int)
    return mse, f1_score(gold, pred, average='weighted')


def hide_edges(G, E, vis, fraction=.1):
    """hide some edges involving hidden node without deconnecting the graph"""
    degrees = {u: sum(adj.values()) for u, adj in G.items()}
    h_degrees = heap({u: -sum(adj.values()) for u, adj in G.items()
                      if not vis[u]})
    edges_hidden = set()
    num_edges_to_hid = int(fraction*len(E))
    while h_degrees and len(edges_hidden) < num_edges_to_hid:
        u = h_degrees.pop()
        neighbors = sorted((v for v in G[u]
                            if degrees[v] > max(2, .7*len(G[v]))),
                           key=degrees.__getitem__, reverse=True)
        for v in neighbors:
            degrees[u] -= 1
            degrees[v] -= 1
            if v in h_degrees:
                h_degrees[v] += 1
            edges_hidden.add((u, v) if u < v else (v, u))
            if len(edges_hidden) >= num_edges_to_hid:
                break
    return edges_hidden


def construct_optimization_matrix(users, xs, edges, visibility, user_order,
                                  weights):
    d = users.shape[1]
    nn = len(user_order)*d
    Q = np.zeros((nn, nn))
    c = np.zeros((nn, 1))
    for i, j, k in ((u, v, k) for (u, v, dirs) in edges.items()
                    for k in dirs):
        if i > j:
            i, j = j, i
        oi, oj = i, j
        ui = users[oi, :]
        uj = users[oj, :]
        xij = xs[k, :]
        i_known = visibility[i]
        j_known = visibility[j]
        i_unknown = not i_known
        j_unknown = not j_known
        w = weights[k]
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
                Q[d*i+k, d*j+l] += w*-xij[k]*xij[l]
        if i_unknown:
            # add ui.xij ² contribution
            for k, l in combinations(range(d), 2):
                Q[d*i+k, d*i+l] += w*xij[k]*xij[l]
            for k in range(d):
                Q[d*i+k, d*i+k] += w*xij[k]*xij[k]/2
        if j_unknown:
            # add uj⋅xij ² contribution
            for k, l in combinations(range(d), 2):
                Q[d*j+k, d*j+l] += w*xij[k]*xij[l]
            for k in range(d):
                Q[d*j+k, d*j+k] += w*xij[k]*xij[k]/2
        if i_known:
            # add cross term when u_i is constant
            ui_xij = w*np.dot(ui, xij)
            c[d*j:d*(j+1)] += np.reshape(-2*xij.T*ui_xij, (d, 1))
        if j_known:
            # add cross term when u_j is constant
            uj_xij = w*np.dot(uj, xij)
            c[d*i:d*(i+1)] += np.reshape(-2*xij.T*uj_xij, (d, 1))
    return Q.T + Q, c
