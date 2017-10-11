import random
from collections import defaultdict, deque
from enum import Enum
from itertools import combinations, product
from timeit import default_timer as clock

import autograd.numpy as anp
import networkx as nx
import numpy as np
import tqdm
from autograd import grad
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI

seed = 61
random.seed(seed)
prng = np.random.RandomState(seed)
k = 7
nb_dirs = 3
pairs = list(combinations(range(k), nb_dirs))
km = KMeans(k, n_init=14, random_state=prng, n_jobs=-2)
d = 35
nrep = 600
blocks = list()
VVar = Enum('VVar', 'W a'.split())
vecc = None


def disjoint_union_all(graphs):
    H = graphs[0].copy()
    for g in graphs[1:]:
        n = len(H)
        H.add_edges_from(((u+n, v+n) for u, v in g.edges()))
    return H


def connect_graph(g):
    components = [list(c) for c in sorted(nx.connected_components(g), key=len, reverse=True)]
    h1 = components[0]
    for h2 in components[1:]:
        g.add_edge(random.choice(h1), random.choice(h2))


def find_failed(rG, labels, adj):
    new_failed = []
    for u, nei in rG.items():
        possible_labels = set(pairs)
        for v in nei:
            possible_labels &= adj[labels[v]]
        if labels[u] not in possible_labels:
            new_failed.append(u)
    return new_failed


def create_random_labeling(rG, adj):
    node = random.choice(list(rG))
    labels = {}
    queue = deque([node, ])
    while queue:
        v = queue.popleft()
        if v in labels:
            continue
        possible_labels = set(pairs)
        for w in rG[v]:
            if w in labels:
                possible_labels &= adj[labels[w]]
            else:
                queue.append(w)
        if possible_labels:
            labels[v] = random.choice(list(possible_labels))
        else:
            labels[v] = random.choice(pairs)
    return labels


def create_blocks(nb_blocks, n, adj):
    # blocks = []
    labeling = []
    for i in range(nb_blocks):
        # blocks.append(nx.fast_gnp_random_graph(n, 4 / n, seed=seed + i))
        # connect_graph(blocks[-1])
        rG = nx.to_dict_of_lists(blocks[i])
        best_labels, min_fail = None, 2e9
        nb_iter = 0
        while nb_iter < 7000 and min_fail > 0:
            nb_iter += 1
            labels = create_random_labeling(rG, adj)
            new_failed = find_failed(rG, labels, adj)
            if len(new_failed) < min_fail:
                best_labels, min_fail = dict(labels), len(new_failed)
        labeling.append(dict(best_labels))
        # print(i, nb_iter, min_fail)
    return blocks, labeling


def assemble_blocks(blocks, labeling, adj):
    assert all((len(b) == len(blocks[0]) for b in blocks))
    n = len(blocks[0])
    available_nodes_by_type = [defaultdict(set) for _ in blocks]
    for i, (g, labels) in enumerate(zip(blocks, labeling)):
        for u in g:
            for possible_label in adj[labels[u]]:
                available_nodes_by_type[i][possible_label].add(u)
    possible_inter_edges = defaultdict(list)
    for i, j in combinations(range(len(blocks)), 2):
        g, labels = blocks[i], labeling[j]
        for u in g:
            u_label = labeling[i][u]
            possible_inter_edges[(i, j)].extend(((n * i + u, n * j + v)
                                                 for v in available_nodes_by_type[j][u_label]))
    G = disjoint_union_all(blocks)
    labels = {i * n + u: label for i, labels in enumerate(labeling)
              for u, label in labels.items()}
    # print([labels[i] for i in range(10)])
    linking_edges = []
    nb_edges_between_blocks = int(G.number_of_edges() / 3 / len(possible_inter_edges))
    for edges in possible_inter_edges.values():
        linking_edges.extend(random.sample(edges, nb_edges_between_blocks))
    G.add_edges_from(linking_edges)
    return G, labels


def create_full_graph(nb_blocks=4, n=100):
    def intersect(a, b):
        return len(set(a) & set(b)) >= 1
    adj = {}
    for p in pairs:
        nei = {q for q in pairs if q != p and intersect(p, q)}
        adj[p] = nei
        # adj[p].add(p)
    blocks, labeling = create_blocks(nb_blocks, n, adj)
    G, labels = assemble_blocks(blocks, labeling, adj)

    mapping = {v: i for i, v in enumerate(sorted(G))}
    inv_mapping = {v: k for k, v in mapping.items()}
    # H = G.copy()
    # nx.relabel_nodes(H, mapping, copy=False)
    num_failed = len(find_failed(nx.to_dict_of_lists(G), labels, adj))
    # print(H.number_of_nodes(), H.number_of_edges(), num_failed)
    return G, labels, mapping, inv_mapping, num_failed


def assign_edges(H, labels, inv_mapping):
    E = {}
    ambiguous = 0
    for u, v in H.edges():
        nu, nv = inv_mapping[u], inv_mapping[v]
        pl = list(set(labels[nu]) & set(labels[nv]))
        ambiguous += labels[nu] == labels[nv]
        E[(u, v)] = random.choice(pl)
    edges, wc = [], []
    for e, w in sorted(E.items()):
        edges.append(e)
        wc.append(w)
    return E, np.array(edges), np.array(wc)


def generate_W(k, d, nb_overlap=0):
    from math import ceil
    available_positions = []
    all_pos = list(range(d))
    for i in range(int(ceil((d + nb_overlap) / d))):
        random.shuffle(all_pos)
        available_positions.extend(list(all_pos))
    nz_positions = [[] for _ in range(k)]
    assigned = 0
    for i, pos in enumerate(available_positions):
        if pos not in nz_positions[i % k]:
            assigned += 1
            nz_positions[i % k].append(pos)
            if assigned == d + nb_overlap:
                break
    W = []
    for zpos in nz_positions:
        zpos.sort()
        w = np.zeros(d)
        w[zpos] = prng.uniform(2, 6, len(zpos))
        for p in zpos:
            if random.random() > .5:
                w[p] *= -1
        W.append(w)
    W = np.array(W)
    return W / np.sqrt((W**2).sum(1))[:, np.newaxis]


def initial_profiles(H, labels, mapping, W):
    profiles = np.zeros((len(H), d))
    for u, dirs in labels.items():
        if u not in mapping:
            continue
        ws = [W[d, :] for d in dirs]
        w = np.sum(ws, 0)
        for wi in ws:
            if random.random() > 1/len(ws):
                w[np.abs(wi) > .1] *= np.sign(wi)[np.abs(wi) > .1]
        profiles[mapping[u], :] = w / np.linalg.norm(w)
    return profiles


def edges_score(u):
    U = anp.reshape(u, (n, d))
    m = U[edges[:, 0]] * U[edges[:, 1]]
    return anp.einsum('ij,ji->i', m, ((1 + mu / k) * W[wc, :].T - vecc[:, np.newaxis])).mean()


def edges_score_max(x0, learning_rate=1, max_iter=50):
    global vecc
    costs = []
    amis = []
    x = np.copy(x0)
    vecc = (mu / k) * W.sum(0)
    all_X = []
    for i in range(max_iter):
        f, g = edges_score(x), edges_score_grad(x)
        costs.append(f)
        m = x[edges[:, 0]] * x[edges[:, 1]]
        ami = AMI(wc, np.argmax(m@W.T, 1))
        amis.append(ami)
        x += learning_rate * g
        norms = np.sqrt((x**2).sum(1))
        x /= norms[:, np.newaxis]
        all_X.append(np.copy(x))
    return x, costs, amis, all_X[find_good_tradeoff(costs, amis)]


def minimize_u_distance_to_incoming(H, E, U0, W):
    rH = nx.to_dict_of_lists(H)
    nU = np.copy(U0)
    nodes = list(rH)
    random.shuffle(nodes)
    stats = np.zeros((len(rH), 8))
    for u in nodes:
        nei = rH[u]
        local_E, local_wc = [], []
        w = 0
        for v in nei:
            nu, nv = (u, v) if u < v else (v, u)
            local_E.append((nu, nv))
            local_wc.append(E[(nu, nv)])
            w += W[local_wc[-1], :]
        nw = w / np.linalg.norm(w)

        def m(x): return np.array([x * nU[e[0] if e[0] != u else e[1], :] for e in local_E])
        x = np.copy(U0[u, :])
        prev_x = np.copy(x)
        score = (np.argmax(m(x)@W.T, 1) == local_wc).sum()
        if (np.argmax(m(nw)@W.T, 1) == local_wc).sum() >= score:
            nU[u, :] = np.copy(nw)
            continue
        new_score = score
        f, g = ((x - w)**2).sum(), 2 * (x - w)
        nb_iter = 0
        eps = .1
        costs = [f, ]
        while nb_iter < 100 and new_score >= score:
            prev_x = np.copy(x)
            x -= eps * g
            x /= np.linalg.norm(x)
            f, ng = ((x - w)**2).sum(), 2 * (x - w)
            new_score = (np.argmax(m(x)@W.T, 1) == local_wc).sum()
            costs.append(f)
            g = ng
            nb_iter += 1
        if new_score < score:
            x = np.copy(prev_x)
        stats[u, :] = (np.linalg.norm(U0[u, :] - nw), np.linalg.norm(x - nw), nb_iter,
                       np.max(m(U0[u, :])@W.T, 1).sum(), np.max(m(x)@W.T, 1).sum(),
                       np.linalg.norm(U0[u, :] - w), np.linalg.norm(x - w), np.linalg.norm(nw - w))
        nU[u, :] = np.copy(x)
    return nU, stats


def compute_hard_cost_l1(all_w, vec_edges):
    dot_prod = vec_edges@(all_w.T)
    return dot_prod.max(1).sum() / vec_edges.shape[0]


def soft_cost_and_grad_matform(vec_edges, W, a, eta=700):
    scores = vec_edges@W.T
    ne = scores.shape[0]
    expo = np.exp(eta * scores)
    denom = expo.sum(1)
    f = (a * np.log(denom)).sum() / (ne * eta)
    g_a = np.log(denom) / (ne * eta)
    expo = np.exp(eta * (scores - scores.max()))
    denom = expo.sum(1)
    g_W = expo.T@(a[:, np.newaxis] * vec_edges / denom[:, np.newaxis])
    return f, g_W / ne, g_a


def vec_max_edge_score(vec_edges, x0, a0, alpha=1e-4, max_iter=250):
    costs_ = []
    rcosts = []
    x = np.copy(x0)
    for nb_iter in range(max_iter):
        f, g, _ = soft_cost_and_grad_matform(vec_edges, x, a0, 500)
        try:
            int(f)
        except (ValueError, OverflowError):
            return costs_, rcosts, x
        x += alpha * g
        x /= np.maximum(1.0, np.sqrt((x**2).sum(1)))[:, np.newaxis]
        costs_.append(f)
        rcosts.append(compute_hard_cost_l1(x, vec_edges))
    return costs_, rcosts, x


def vec_node_loss_wrt_w(w):
    x = anp.reshape(w, W.shape)
    dp = anp.dot(m, x.T)
    e = anp.exp(eta * dp)
    P = e / e.sum(1)[:, anp.newaxis]
    wij = P@W
    aw = a[:, anp.newaxis] * wij
    nn = Bd.shape[0]
    v = U - ideg * anp.dot(Bd, aw)
    return (v**2).sum() / nn


def vector_mixed(D, W0, a0, to_optimize, node_factor, num_loop=3, inner_size=25, ef=1):
    costs, costs_edge, costs_node = [], [], []
    xw, xa = np.copy(W0), np.copy(a0)
    for loop, var, i in product(range(num_loop), to_optimize, range(inner_size)):
        c_node = vec_node_loss_wrt_w(xw.ravel())
        costs_node.append(c_node)
        c_edge, g_edge_w, g_edge_a = soft_cost_and_grad_matform(D, xw, xa, 700)
        costs_edge.append(c_edge)
        costs.append(c_edge + node_factor * c_node)
        alpha = {VVar.W: 1e-1,
                 VVar.a: 1e-2}[var]
        if var == VVar.W:
            if node_factor > 0:
                g_node_w = np.reshape(grad_vec_node_loss_wrt_w(xw.ravel()), xw.shape)
            else:
                g_node_w = 0
            xw -= alpha * (node_factor * g_node_w - ef * g_edge_w)
            norms = np.sqrt((xw**2).sum(1))
            xw /= np.maximum(1.0, norms)[:, np.newaxis]
    return xw, xa, np.array(costs_node), np.array(costs_edge)


if __name__ == "__main__":
    import time
    all_res = np.zeros((nrep, 10))
    timestamp = (int(time.time()-(2017-1970)*365.25*24*60*60))//60
    suffix = 'GfixedWvariable0over_kminit_ambiguous_3dir_7w'
    # W = generate_W(k, d, 0)
    num_nodes, block_size = 600, 125
    nb_blocks = np.floor_divide(num_nodes, block_size)
    block_size = num_nodes // nb_blocks
    for i in range(nb_blocks):
        blocks.append(nx.fast_gnp_random_graph(block_size, 4 / block_size))
        connect_graph(blocks[-1])
    H, labels, mapping, inv_mapping, num_failed = create_full_graph(nb_blocks, block_size)
    for rep in tqdm.trange(nrep, unit='run'):
    # for rep in range(nrep):
        start = clock()
        W = generate_W(k, d, 0)
        # H, labels, mapping, inv_mapping, num_failed = create_full_graph(4, 125)
        if num_failed > 0:
            continue
        E, edges, wc = assign_edges(H, labels, inv_mapping)
        n = len(H)
        U = initial_profiles(H, labels, mapping, W)
        m = U[edges[:, 0]] * U[edges[:, 1]]
        all_res[rep, (0, 1)] = (AMI(wc, np.argmax(m@W.T, 1)),
                                AMI(wc, km.fit_predict(m/np.sqrt((m**2).sum(1))[:, np.newaxis])))
        # print('[{:.3f}]\tGenerated graph and initial profiles'.format(clock()-start)); start=clock()

        U0 = np.copy(U)
        edges_score_grad = grad(edges_score)
        mu, alpha, T = 0, 8e2, 15
        _, c, am_nomin, xU = edges_score_max(U0, alpha, T)
        m = xU[edges[:, 0]] * xU[edges[:, 1]]
        all_res[rep, (2, 3)] = (AMI(wc, np.argmax(m@W.T, 1)),
                                AMI(wc, km.fit_predict(m/np.sqrt((m**2).sum(1))[:, np.newaxis])))
        # print('[{:.3f}]\tFirst optimization of profiles ({:.3f})'.format(clock()-start, all_res[rep, 2])); start=clock()
        # continue

        # x0 = prng.randn(*W.shape)
        # x0 /= np.maximum(1.0, np.sqrt((x0**2).sum(1)))[:, np.newaxis]
        vec_edges = m
        # cost, rcost, xw = vec_max_edge_score(m, x0, np.ones(m.shape[0]), 5e1, 50)
        cost, rcost, xw = vec_max_edge_score(m, np.copy(km.cluster_centers_), np.ones(m.shape[0]), 5e1, 20)
        all_res[rep, 4] = AMI(wc, np.argmax(m@xw.T, 1))
        all_res[rep, 8] = np.mean(cdist(W, xw).min(0))
        # np.savez_compressed('recover_waldo_{}_{}_{}'.format(timestamp, seed, suffix), all_res=all_res)
        # continue
        # print('[{:.3f}]\tTried to recover W from there'.format(clock()-start)); start=clock()

        nU, _ = minimize_u_distance_to_incoming(H, E, xU, W)
        m = nU[edges[:, 0]] * nU[edges[:, 1]]
        all_res[rep, (5, 6)] = (AMI(wc, np.argmax(m@W.T, 1)),
                                AMI(wc, km.fit_predict(m/np.sqrt((m**2).sum(1))[:, np.newaxis])))
        # print('[{:.3f}]\tSecond optimization of profiles'.format(clock()-start)); start=clock()

        U = nU
        a = np.ones(m.shape[0])
        B = nx.incidence_matrix(H)
        Bd = B.toarray().astype(np.int8)
        D = 1 / np.array(B.sum(1)).ravel()
        eta = 700
        ideg = D[:, np.newaxis]
        grad_vec_node_loss_wrt_w = grad(vec_node_loss_wrt_w)
        xw, xa, cn, ce = vector_mixed(m, np.copy(km.cluster_centers_), np.ones(m.shape[0]), [VVar.W, ],
                                      node_factor=1, num_loop=1, inner_size=40, ef=.8)
        all_res[rep, 7] = AMI(wc, np.argmax(m@xw.T, 1))
        # print('[{:.3f}]\tTried to recover W from there'.format(clock()-start)); start=clock()

        # print(' '.join(['{:.4f}'.format(v) for v in all_res[rep, :]]))
        all_res[rep, 9] = np.mean(cdist(W, xw).min(0))
        np.savez_compressed('recover_waldo_{}_{}_{}'.format(timestamp, seed, suffix), all_res=all_res)
