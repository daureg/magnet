import random
from collections import defaultdict, deque
from enum import Enum
from itertools import combinations, product

import autograd.numpy as anp
import networkx as nx
import numpy as np
import scipy.sparse as sparse
import torch
import tqdm
from autograd import grad
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI
from torch import autograd, nn, optim

seed = 66
random.seed(seed)
prng = np.random.RandomState(seed)
k = 7
nb_dirs = 3
d = 35
nrep = 200
blocks = list()
VVar = Enum('VVar', 'W a'.split())
Var = Enum('Var', 'P Q a'.split())
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


def pytorch_optim(W, edges, wc, n, alpha=1, beta=1, max_iter=50, x0=None, lr=1):
    d = W.shape[1]
    X = prng.randn(n, d) if x0 is None else np.copy(x0)
    tX = autograd.Variable(torch.from_numpy(X), requires_grad=True)
    tW = autograd.Variable(torch.from_numpy(W), requires_grad=False)
    target = autograd.Variable(torch.from_numpy(wc), requires_grad=False)
    head = autograd.Variable(torch.from_numpy(edges[:, 0]), requires_grad=False)
    tail = autograd.Variable(torch.from_numpy(edges[:, 1]), requires_grad=False)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam([tX], weight_decay=0, lr=lr)
    l_res = np.zeros((max_iter + 1, 4))
    nX = X / np.sqrt((X**2).sum(1))[:, np.newaxis]
    m = nX[edges[:, 0]] * nX[edges[:, 1]]
    scores = m@W.T
    l_res[0, 3] = np.einsum('ij,ji->i', m, W[wc, :].T).mean()
    pred = np.argmax(scores, 1)
    l_res[0, 1] = AMI(wc, pred)
    S = tX[head] * tX[tail]
    output = torch.mm(S, torch.t(tW))
    loss = loss_fn(output, target)
    l_res[0, 0] = (loss_fn(output, target).data[0])
    for i in range(max_iter):
        optimizer.zero_grad()
        S = tX[head] * tX[tail]
        output = torch.mm(S, torch.t(tW))
        # avg_norm = torch.mean(torch.norm(tX, p=2, dim=1))
        # avg_edge = torch.mean(torch.diag(torch.mm(S, torch.t(tW[wc, :]))))
        loss = loss_fn(output, target)  # - alpha*avg_edge# + beta*avg_norm
        l_res[i + 1, 0] = (loss_fn(output, target).data[0])
        loss.backward()
        optimizer.step()
        nX = X / np.sqrt((X**2).sum(1))[:, np.newaxis]
        m = nX[edges[:, 0]] * nX[edges[:, 1]]
        scores = m@W.T
        l_res[i + 1, 3] = np.einsum('ij,ji->i', m, W[wc, :].T).mean()
        pred = np.argmax(scores, 1)
        l_res[i + 1, 1] = AMI(wc, pred)
    # print(np.sqrt((X**2).sum(1)).mean())
    return nX, l_res[:i + 2, :]


def pareto_front(costs):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    from https://stackoverflow.com/a/40239615
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Remove dominated points
            is_efficient[is_efficient] = np.any(costs[is_efficient] >= c, axis=1)
    return is_efficient


def find_good_tradeoff(edge_scores, ami_scores):
    scores = np.array((edge_scores, ami_scores)).T
    pf = pareto_front(scores)
    top_right_corner = np.array((max(edge_scores), max(ami_scores)))[np.newaxis, :]
    dst = cdist(scores[pf], top_right_corner, 'mahalanobis')
    return np.where(pf)[0][np.argmin(dst)]


def finetune(x0, nb_iter=10):
    xU = np.copy(x0)
    m = xU[edges[:, 0]] * xU[edges[:, 1]]
    pred = np.argmax(m@W.T, 1)
    error = np.where(pred != wc)[0]
    initial_error = len(error)
    res = []
    step_size = .04
    bestX, min_error = None, 2 * initial_error
    for s in range(nb_iter):
        m = xU[edges[:, 0]] * xU[edges[:, 1]]
        pred = np.argmax(m@W.T, 1)
        error = np.where(pred != wc)[0]
        res.append(len(error))
        errors = list(error)
        if len(errors) < min_error:
            bestX, min_error = np.copy(xU), len(errors)
        random.shuffle(errors)
        for eid in errors:
            u, v = edges[eid, :]
            xu, xv = xU[u, :], xU[v, :]
            right_w = wc[eid]
            wrong_w = pred[eid]
            c = step_size * (right_w - wrong_w) / (.9 * s + 1)
            xu += xv * c
            xv += xu * c
            xU[u, :] = xu / np.sqrt((xu**2).sum())
            xU[v, :] = xv / np.sqrt((xv**2).sum())
    m = xU[edges[:, 0]] * xU[edges[:, 1]]
    pred = np.argmax(m@W.T, 1)
    error = np.where(pred != wc)[0]
    res.append(len(error))
    return bestX, np.array(res)


def frank_wolfe(S, W=None, mu=0.13, max_nuc=85):
    W = np.random.randn(d, len(S))
    # print(np.linalg.matrix_rank(W), np.linalg.norm(W, 'nuc'),
    #       np.percentile(np.sqrt((W**2).sum(0)), [25, 50, 75]))
    fval = []
    for t in range(k):
        # the other term is mu*np.einsum('ij,ji->i', W.T, W).sum())
        fval.append(-np.einsum('ij,ji->i', S, W).sum())
        g = 2 * mu * W - S.T
        u, s, vt = svds(-g, 1)
        atom = max_nuc * np.outer(u, vt)
        gamma = 2 / (2 + t)
        W += gamma * (atom - W)  # ie w = (1-γ)w + γ⋅atom

    fval.append(-np.einsum('ij,ji->i', S, W).sum())
    return W, fval


def lloyd_heuristic(init_w, max_iter=10):
    km_w = np.copy(init_w)
    km_w /= np.maximum(1.0, np.sqrt((km_w**2).sum(1)))[:, np.newaxis]
    km_labels = km.labels_
    # km_labels = np.argmax(m@km_w.T, 1) # how are those two differents? yes widely so
    stats = np.zeros((max_iter, 1))
    for it in range(max_iter):
        stats[it, :] = AMI(wc, km_labels)
        km_labels = np.argmax(m@km_w.T, 1)
        new_W = []
        for i in range(k):
            nw = m[km_labels == i, :].sum(0)
            nw[np.argsort(np.abs(nw))[:-nnz]] = 0
            nw /= np.linalg.norm(nw)
            new_W.append(nw)
        km_w = np.array(new_W)
    return km_w, stats


def matrix_mixed(M, P0, Q0, a0, to_optimize, node_factor, num_loop=3, inner_size=25, ef=1,
                 sum_Q_to_one=False):
    # TODO: xU (was u before) is the user profile, should it be an argument of that function?
    costs, costs_edge, costs_node = [], [], []
    xp, xq, xa = np.copy(P0), np.copy(Q0), np.copy(a0)
    if sum_Q_to_one:
        def normalization_factor(Q): return Q.sum(1)[..., np.newaxis]
    else:
        def normalization_factor(Q): return np.sqrt((Q ** 2).sum(1))[..., np.newaxis]
    xq /= normalization_factor(xq)
    nn, ne = B.shape
    for loop, var, i in product(range(num_loop), to_optimize, range(inner_size)):
        if loop == num_loop - 1 and var == Var.Q:
            break
        if loop == num_loop - 1 and var == Var.P:
            node_factor *= 1
        aA = xa[:, np.newaxis]
        aQ = aA * xq
        f = xU - C - ideg * (B@aQ@xp.T)
        c_node = (f**2).sum() / nn
        costs_node.append(c_node)
        W = xq@xp.T
        R = np.einsum('ij,ji->i', M, W.T * aA.T)
        c_edge = R.sum() / ne
        costs_edge.append(c_edge)
        costs.append(c_edge - node_factor * c_node)
        alpha = {Var.P: 3e-1,
                 Var.Q: 1e-1,
                 Var.a: 8e1}[var]
        if var == Var.P:
            xp -= alpha * (node_factor * (-2 * f.T@(ideg * (B@aQ))) / nn - ef*(aA.T * M.T @ xq) / ne)
            norms = np.sqrt((xp**2).sum(0))
            xp /= norms[np.newaxis, :]
        if var == Var.Q:
            xq -= alpha * (node_factor * (-2 * aA * (invdB.T@f@xp)) / nn - ef*(aA * (M @ xp)) / ne)
            xq[xq < 0] = 0
            xq /= normalization_factor(xq)
        if var == Var.a:
            xa -= alpha * (node_factor * (-2 * xq * (invdB.T@f@xp)).sum(1) / nn - ef*(M.T * W.T).sum(0) / ne)
            xa[xa < 0] = 0
    return xp, xq, xa, np.array(costs_node), np.array(costs_edge)


def msg(s, color='yellow'):
    pass
    # fancy_print(s, color=color, time=True)


if __name__ == "__main__":
    import time
    from fprint import fancy_print
    import argparse
    timestamp = (int(time.time()-(2017-1970)*365.25*24*60*60))//60
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dim", help="dimension of profiles", type=int, default=35)
    parser.add_argument("-o", "--over", help="overlap in profiles", type=int, default=0)
    parser.add_argument("-l", "--local", help="number of dir per node", type=int, default=3)
    parser.add_argument("-k", "--ndir", help="total number of directions", type=int, default=7)
    args = parser.parse_args()
    k, d, n_overlap, nb_dirs = args.ndir, args.dim, args.over, args.local
    assert nb_dirs <= k
    assert (d % k) == 0
    assert n_overlap <= d
    pairs = list(combinations(range(k), nb_dirs))
    km = KMeans(k, n_init=14, random_state=prng, n_jobs=-2)
    conf = '{}over_{}dir_{}w_{}dim'.format(n_overlap, nb_dirs, k, d)
    suffix = 'GfixedWvariable_' + conf

    msg('creating & coloring the graph…', 'blue')
    num_nodes, block_size = 500, 125
    nb_blocks = np.floor_divide(num_nodes, block_size)
    block_size = num_nodes // nb_blocks
    for i in range(nb_blocks):
        blocks.append(nx.fast_gnp_random_graph(block_size, 4 / block_size))
        connect_graph(blocks[-1])
    H, labels, mapping, inv_mapping, num_failed = create_full_graph(nb_blocks, block_size)

    msg('generating labels…', 'blue')
    E, edges, wc = assign_edges(H, labels, inv_mapping)
    n = len(H)
    res = np.zeros((nrep, 14))
    for it in tqdm.trange(nrep, unit='W'):
        W = generate_W(k, d, n_overlap)
        U = initial_profiles(H, labels, mapping, W)
        m = U[edges[:, 0]] * U[edges[:, 1]]
        orig_ami_us, orig_ami_km = (AMI(wc, np.argmax(m@W.T, 1)),
                                    AMI(wc, km.fit_predict(m / np.sqrt((m**2).sum(1))[:, np.newaxis])))
        msg('us: {:.3f}\n{} {:.3f}'.format(orig_ami_us, 'km:'.rjust(14), orig_ami_km))
        orig_W = np.copy(W)

        msg('creating profile with cross-entropy…', 'blue')
        U1, l_res = pytorch_optim(W, edges, wc, len(H), alpha=2, max_iter=7, lr=.5)
        mu, alpha, T = 1.5, 15e2, 15
        msg('then augmenting their score and finetuning…', 'blue')
        edges_score_grad = grad(edges_score)
        _, c, am_nomin, xU = edges_score_max(U1, alpha, T)
        xU, eres = finetune(xU)
        m = xU[edges[:, 0]] * xU[edges[:, 1]]
        gold_edge_score = np.einsum('ij,ji->i', m, W[wc, :].T).mean()
        msg('average edge score: {:.4g}'.format(gold_edge_score))
        edge_accuracy = 100 * (1 - eres.min() / len(E))
        msg('correctly classified edges: {:.2f}%'.format(edge_accuracy))
        orig_ami_us, orig_ami_km = (AMI(wc, np.argmax(m@W.T, 1)),
                                    AMI(wc, km.fit_predict(m / np.sqrt((m**2).sum(1))[:, np.newaxis])))
        msg('us: {:.3f}\n{} {:.3f}'.format(orig_ami_us, 'km:'.rjust(14), orig_ami_km))
        res[it, (0, 1)] = orig_ami_us, edge_accuracy

        msg('optimizing only soft edges score…', 'blue')
        vec_edges = m
        km_pred = km.fit_predict(m / np.sqrt((m**2).sum(1))[:, np.newaxis])
        cost, rcost, xw = vec_max_edge_score(m, np.copy(km.cluster_centers_), np.ones(m.shape[0]), 5e1, 20)
        orig_ami_us, orig_ami_km = (AMI(wc, np.argmax(m@xw.T, 1)), AMI(wc, km_pred))
        us_dst = np.mean(cdist(W, xw).min(0))
        msg('us: {:.3f} (mean dst: {:.3f})\n{} {:.3f}'.format(orig_ami_us, us_dst,
                                                              'km:'.rjust(14), orig_ami_km))
        res[it, (2, 3, 4)] = orig_ami_us, us_dst, orig_ami_km

        msg('optimizing soft edges score AND nodes cost…', 'blue')
        a = np.ones(m.shape[0])
        B = nx.incidence_matrix(H)
        Bd = B.toarray().astype(np.int8)
        D = 1 / np.array(B.sum(1)).ravel()
        eta = 700
        ideg = D[:, np.newaxis]
        grad_vec_node_loss_wrt_w = grad(vec_node_loss_wrt_w)
        xw, xa, cn, ce = vector_mixed(m, np.copy(km.cluster_centers_), np.ones(m.shape[0]), [VVar.W, ],
                                      node_factor=.2, num_loop=1, inner_size=50, ef=.8)
        orig_ami_us, us_dst = (AMI(wc, np.argmax(m@xw.T, 1)), np.mean(cdist(W, xw).min(0)))
        msg('us: {:.3f} (mean dst: {:.3f})'.format(orig_ami_us, us_dst))
        text = 'mean edge score changed from {:.3f} to {:.3f}, and mean node cost from {:.3f} to {:.3f}'
        msg(text.format(ce[0], ce[-1], cn[0], cn[-1]))
        res[it, (5, 6)] = orig_ami_us, us_dst

        msg('improving kmean through heuristics…', 'blue')
        nnz = int((np.abs(W) > .1).sum() / k)
        km_w, stats = lloyd_heuristic(km.cluster_centers_)
        us_km_ami, us_km_dst, us_vs_km_ami = (stats[-1, 0], np.mean(cdist(W, km_w).min(0)),
                                              AMI(np.argmax(m@xw.T, 1), np.argmax(m@km_w.T, 1)))
        msg('AMI: {:.3f} (mean dst: {:.3f})'.format(us_km_ami, us_km_dst))
        msg('Is the solution close to our previous one?  {:.3f}'.format(us_vs_km_ami))
        res[it, (7, 8, 9)] = us_km_ami, us_km_dst, us_vs_km_ami

        msg('Frank-Wolfe with L2 regularization…', 'blue')
        fW, fval = frank_wolfe(np.copy(m), W=None, mu=0.13, max_nuc=85)
        rank, nucnorm, col_norms, edge_score = (np.linalg.matrix_rank(fW),  np.linalg.norm(fW, 'nuc'),
                                                np.sqrt((fW**2).sum(0)), -fval[-1] / len(E))
        prc = ', '.join(['{:.3f}'.format(_) for _ in np.percentile(col_norms, [25, 50, 75, 90])])
        fw_ami = AMI(wc, km.fit_predict(fW.T))
        fw_dst = np.mean(cdist(W, km.cluster_centers_).min(0))
        text = 'rank: {}, nuclear norm: {:.3f}, max column norm: {:.3f} ({})'
        msg(text.format(rank, nucnorm, col_norms.max(), prc))
        msg('AMI: {:.3f} (mean dst: {:.3f})'.format(fw_ami, fw_dst))
        res[it, (10, 11, 12)] = fw_ami, fw_dst, edge_score

        msg('Explicit P & Q optimization, starting from previous solution…', 'blue')
        invdB = sparse.csr_matrix(d * Bd)
        C = np.zeros((n, d))
        to_optimize = [Var.P, Var.Q]
        a0 = np.ones(m.shape[0])
        prevP = np.copy(xw.T)
        prevQ = np.zeros((len(wc), k))
        for i, v in enumerate(np.argmax(m@xw.T, 1)):
            prevQ[i, v] = 1
        P_mat, Q_mat, A_mat, cn, ce = matrix_mixed(m, np.copy(prevP), np.copy(prevQ), a0, to_optimize,
                                                   node_factor=.1, num_loop=4, inner_size=80, ef=1,
                                                   sum_Q_to_one=False)
        text = 'mean edge score changed from {:.3f} to {:.3f}, and mean node cost from {:.3f} to {:.3f}'
        msg(text.format(ce[0], ce[-1], cn[0], cn[-1]))
        us_vs_pq_ami = AMI(np.argmax(m@xw.T, 1), km.fit_predict(Q_mat))
        msg('agree with previous solution as: {:.3f}'.format(us_vs_pq_ami))
        res[it, 13] = us_vs_pq_ami
        np.savez_compressed('recover_waldo_{}_{}_{}'.format(timestamp, seed, suffix), res=res,
                            conf=np.array((n_overlap, nb_dirs, k, d)))
