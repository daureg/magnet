# vim: set fileencoding=utf-8
"""
Implement the Shazoo node binary classification algorithm.

from Vitale, F., Cesa-Bianchi, N., Gentile, C., & Zappella, G. (2011).
See the tree through the lines: the Shazoo algorithm.
In Advances in Neural Information Processing Systems 24 (pp. 1584–1592).
http://papers.nips.cc/paper/4476-see-the-tree-through-the-lines-the-shazoo-algorithm .
"""
from __future__ import division
from collections import deque, defaultdict
import convert_experiment as cexp
import random
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import bicgstab
from future.utils import iteritems
from timeit import default_timer as clock
from random_tree import get_tree as get_rst_tree
from grid_stretch import perturbed_bfs as get_bfs_tree
from grid_stretch import add_edge
from new_galaxy import galaxy_maker as get_stg_tree
from multiprocessing import Pool
MAX_WEIGHT = int(2e9)
UNKNOWN, REVEALED, FORK, HINGE = 0, 1, 2, 3
FLEP_CALLS_TIMING = []
GRAPH, TREE_ADJ, TWEIGHTS = None, None, None
EWEIGHTS, SIGNS, VTRAIN = None, None, None


def profile(f):
    return f


def get_rst(fake):
    _, edges, _ = get_rst_tree(GRAPH, EWEIGHTS)
    from_edges_to_tree(edges)


def get_bfs(root):
    from_edges_to_tree(get_bfs_tree(GRAPH, root))


def get_stg(**func_args):
    edges, _ = get_stg_tree(GRAPH, 50, output_name=None, short=True,
                            **func_args)
    from_edges_to_tree(edges)


def from_edges_to_tree(edges):
    global TREE_ADJ, TWEIGHTS
    adj = {}
    for u, v in edges:
        add_edge(adj, u, v)
    TREE_ADJ, TWEIGHTS = adj, {e: EWEIGHTS[e] for e in edges}


def majority_vote(preds):
    """Aggregate all prediction by majority vote."""
    if len(preds) == 1:
        return preds
    return [1 if sum(votes) > 0 else -1
            for votes in zip(*preds)]


def predict(args):
    tree_generation, tree_args = args
    tree_generation(**tree_args)
    return offline_shazoo(TREE_ADJ, TWEIGHTS, SIGNS, VTRAIN)


def run_committee(graph, eweights, signs, tree_kind='rst', train_vertices=.1,
                  size=13, degree_function=None, threshold_function=None):
    global GRAPH, EWEIGHTS, SIGNS, VTRAIN
    GRAPH, EWEIGHTS, SIGNS = graph, eweights, signs
    if isinstance(train_vertices, float):
        num_revealed = int(train_vertices*len(graph))
        train_vertices = random.sample(list(graph.keys()), num_revealed)
    VTRAIN = train_vertices
    tree_kind = tree_kind.lower()
    assert tree_kind in ['rst', 'bfs', 'stg'], tree_kind
    if tree_kind == 'rst':
        args = size*[(get_rst, {'fake': None}), ]
    if tree_kind == 'bfs':
        degrees = sorted(((node, len(adj)) for node, adj in graph.items()),
                         key=lambda x: x[1])
        args = [(get_bfs, {'root': _[0]}) for _ in degrees[-size:]]
    if tree_kind == 'stg':
        func_dict = {'degree_function': degree_function,
                     'threshold_function': threshold_function}
        args = size*[(get_stg, func_dict), ]
    num_threads = min(6, size)
    pool = Pool(num_threads)
    res = list(pool.imap_unordered(predict, args,
                                   chunksize=size//num_threads))
    preds, gold = [_[1] for _ in res], res[0][0]
    return gold, majority_vote(preds)


@profile
def _edge(u, v):
    """Reorder u and v."""
    return (u, v) if u < v else (v, u)


@profile
def flep(tree_adj, nodes_sign, edge_weight, root, return_fullcut_info=False):
    """Compute the sign of the `root` that yield the smallest weighted cut in
    `tree_adj` given the already revealed `nodes_sign`."""
    start = clock()
    assert isinstance(tree_adj, dict)
    if root in nodes_sign:
        cutp, cutn = (MAX_WEIGHT, 0) if nodes_sign[root] < 0 else (0, MAX_WEIGHT)
        return nodes_sign[root]*MAX_WEIGHT, {}, {root: (True, -1, cutp, cutn)}
    assert root not in nodes_sign
    stack = []
    status = defaultdict(lambda: (False, -1, 0, 0))
    stack.append(root)
    while stack:
        v = stack.pop()
        if v >= 0:
            discovered, pred, cutp, cutn = status[v]
        else:
            v = -(v+100)
            discovered, pred, cutp, cutn = status[v]
            for child in tree_adj[v]:
                if status[child][1] != v:
                    continue
                eweight = edge_weight[(child, v) if child < v else (v, child)]
                _, _, childp, childn = status[child]
                cutp += min(childp, childn + eweight)
                cutn += min(childn, childp + eweight)
            status[v] = (discovered, pred, cutp, cutn)
            # print('{}: (+: {}, -: {})'.format(v, cutp, cutn))
            if v == root:
                FLEP_CALLS_TIMING.append(clock() - start)
                intermediate = {}
                if return_fullcut_info:
                    intermediate = {n: (vals[2], vals[3])
                                    for n, vals in status.items()
                                    if vals[0] and n not in nodes_sign}
                return (cutn - cutp, intermediate, status)

        if not discovered:
            status[v] = (True, pred, cutp, cutn)
            if v in nodes_sign:
                # don't go beyond revealed nodes
                continue
            stack.append(-(v+100))
            for w in tree_adj[v]:
                discovered, pred, cutp, cutn = status[w]
                if pred == -1 and w != root:
                    if w in nodes_sign:
                        cutp, cutn = (MAX_WEIGHT, 0) if nodes_sign[w] < 0 else (0, MAX_WEIGHT)
                    status[w] = (discovered, v, cutp, cutn)
                if not discovered:
                    stack.append(w)
    assert False, root


@profile
def is_a_fork(tree_adj, node, hinge_lines):
    """If node has more than 3 hinge edges incident, it's a fork."""
    incident_hinge = 0
    for u in tree_adj[node]:
        incident_hinge += int(hinge_lines[_edge(u, node)])
        if incident_hinge >= 3:
            return True
    return False


@profile
def reveal_node(tree_adj, node, nodes_status, hinge_lines, ancestors):
    """Upon `node` sign revelation, traverse the tree to update the status of
    its nodes and edges."""
    nodes_status[node] = REVEALED
    parent = ancestors[node]
    while parent is not None:
        edge = _edge(node, parent)
        hinge_lines[edge] = True
        potential_fork = None
        if nodes_status[parent] == UNKNOWN:
            nodes_status[parent] = HINGE
        elif nodes_status[parent] == REVEALED:
            potential_fork = node
        elif nodes_status[parent] == HINGE:
            potential_fork = parent
        elif nodes_status[parent] == FORK:
            potential_fork = node
        if potential_fork is not None and \
           nodes_status[potential_fork] != REVEALED:
            if is_a_fork(tree_adj, potential_fork, hinge_lines):
                assert nodes_status[potential_fork] != REVEALED
                nodes_status[potential_fork] = FORK
        if nodes_status[parent] in [REVEALED, FORK]:
            break
        node, parent = parent, ancestors[parent]


@profile
def predict_node_sign(tree_adj, node, nodes_status, nodes_sign, hinge_lines,
                      edge_weight):
    """Perform one step (ie prediction) of the Shazoo algorithm.

    We traverse the `tree_adj` (rooted at `node`) in a BFS manner in order to
    find the closest connection node (in the sense of the resistance distance)
    and compute its best sign.
    """
    q = deque()
    status = defaultdict(lambda: (False, 0))
    q.append(node)
    status[node] = (True, 0)
    connect_nodes = {}
    min_connect, min_connect_distance = None, 2e9
    while q:
        v = q.popleft()
        distance_from_root = status[v][1]
        v_status = nodes_status[v]
        if v_status == REVEALED:
            connect_nodes[v] = nodes_sign[v]
            if distance_from_root < min_connect_distance:
                min_connect, min_connect_distance = v, distance_from_root
            continue
        if v_status == FORK:
            assert v not in nodes_sign, (v, v_status)
            estim, _ = flep(tree_adj, nodes_sign, edge_weight, v)
            if abs(estim) > 1e-4:
                connect_nodes[v] = 1 if estim > 0 else -1
                if distance_from_root < min_connect_distance:
                    min_connect, min_connect_distance = v, distance_from_root
            continue
        if distance_from_root >= min_connect_distance:
            continue
        for w in tree_adj[v]:
            edge = _edge(v, w)
            if not status[w][0]:
                q.append(w)
                status[w] = (True, distance_from_root + 1/edge_weight[edge])

    # print(connect_nodes, min_connect, min_connect_distance)
    return -1 if min_connect is None else connect_nodes[min_connect]


def make_graph(n, tree=True):
    if tree:
        cexp.fast_preferential_attachment(n, 1)
    else:
        cexp.fast_preferential_attachment(n, 3, .13)
    ci = cexp.turn_into_signed_graph_by_propagation(6, infected_fraction=0.9)
    adj = cexp.redensify.G
    ew = {e: 12*random.random() for e in cexp.redensify.EDGES_SIGN}

    def to_bin_sign(val):
        val = 1 if val % 2 == 0 else -1
        return val if random.random() > .04 else -val
    gold_sign = {i: to_bin_sign(v) for i, v in enumerate(ci)}
    nodes_status = {n: UNKNOWN for n in adj}
    hinge_lines = {e: False for e in ew}
    nodes_sign = {}
    num_phi = sum((1 for e in ew
                   if gold_sign[e[0]] != gold_sign[e[1]]))
    print('phi edges: {}'.format(num_phi))
    return (adj, nodes_status, ew, hinge_lines, nodes_sign, gold_sign), num_phi


@profile
def shazoo(tree_adj, nodes_status, edge_weight, hinge_lines, nodes_sign,
           gold_sign):
    """Predict all the signs of `gold_sign` in an online fashion."""
    from grid_stretch import ancestor_info
    order = list(gold_sign.keys())
    random.shuffle(order)
    allpred = {}
    node = order[0]
    nodes_sign[node] = gold_sign[node]
    nodes_status[node] = REVEALED  # no need for full reveal call
    ancestors = ancestor_info(tree_adj, node)
    allpred[node] = -1
    for node in order[1:]:
        pred = predict_node_sign(tree_adj, node, nodes_status, nodes_sign,
                                 hinge_lines, edge_weight)
        allpred[node] = pred
        nodes_sign[node] = gold_sign[node]
        reveal_node(tree_adj, node, nodes_status, hinge_lines, ancestors)
    mistakes = sum((1 for n, p in allpred.items() if p != gold_sign[n]))
    print('mistakes: {}'.format(mistakes))


@profile
def offline_cut_computation(tree_adj, nodes_sign, edge_weight, root):
    _, rooted_cut = flep(tree_adj, nodes_sign, edge_weight, root,
                         return_fullcut_info=True)
    queue = deque()
    discovered = {k: k == root for k in tree_adj}
    queue.append(root)
    while queue:
        v = queue.popleft()
        if v in nodes_sign:
            continue
        for u in tree_adj[v]:
            if not discovered[u]:
                queue.append(u)
                discovered[u] = True
                if u in nodes_sign:
                    continue
                u_cp, u_cn = rooted_cut[u]
                p_cp, p_cn = rooted_cut[v]
                ew = edge_weight[_edge(u, v)]
                no_child_cp = p_cp - min(u_cp, u_cn + ew)
                no_child_cn = p_cn - min(u_cn, u_cp + ew)
                rooted_cut[u] = (u_cp + min(no_child_cp, no_child_cn + ew),
                                 u_cn + min(no_child_cn, no_child_cp + ew))
    return rooted_cut


def offline_shazoo(tree_adj, edge_weights, node_signs, train_vertices):
    test_vertices = set(tree_adj.keys()) - set(train_vertices)
    nodes_status = {n: UNKNOWN for n in tree_adj}
    preds = {}
    seen_signs = {}
    hinge_lines = {e: False for e in edge_weights}

    from grid_stretch import ancestor_info
    node = train_vertices[0]
    nodes_status[node] = REVEALED  # no need for full reveal call
    seen_signs[node] = node_signs[node]
    ancestors = ancestor_info(tree_adj, node)
    for node in train_vertices[1:]:
        seen_signs[node] = node_signs[node]
        reveal_node(tree_adj, node, nodes_status, hinge_lines, ancestors)
    for node in test_vertices:
        if node in preds:
            continue
        cuts = offline_cut_computation(tree_adj, seen_signs, edge_weights,
                                       node)
        new_pred = {n: 1 if cut[0] < cut[1] else -1
                    for n, cut in cuts.items()}
        assert all([n not in preds for n in new_pred])
        preds.update(new_pred)
    pred, gold = [], []
    assert set(preds.keys()) == test_vertices
    for node, sign in sorted(preds.items()):
        pred.append(sign)
        gold.append(node_signs[node])
    return gold, pred


@profile
def find_hinge_nodes(tree_adj, edge_weight, nodes_sign, node_to_predict,
                     with_distances=False):
    """Find hinge nodes in `tree_adj` when trying to predict one node sign.

    This implements the first step of the online shazoo algorithm.
    """
    stack = []
    # the fields are: discovered, distance_from_root, incident_hinge_line, marked
    status = defaultdict(lambda: [False, float(MAX_WEIGHT), 0, False])
    stack.append(node_to_predict)
    status[node_to_predict] = [False, 0.0, 0, False]
    while stack:
        v = stack.pop()
        if v >= 0:
            discovered, distance_from_root, incident_hinge_line, marked = status[v]
        else:
            v = -(v + 100)
            discovered, distance_from_root, _, _ = status[v]
            update_mark_status(v, tree_adj, status, distance_from_root)
            if v == node_to_predict:
                if status[node_to_predict][2] == 1 and len(nodes_sign) > 1:
                    clean_root_hinge(node_to_predict, tree_adj, status)
                return extract_hinge_nodes(status, nodes_sign, with_distances)
        if not discovered:
            status[v][0] = True
            if v in nodes_sign:
                # mark revealed nodes but don't go beyond them
                status[v][3] = True
                continue
            stack.append(-(v+100))
            for w in tree_adj[v]:
                if status[w][0]:
                    continue
                weight = edge_weight[(v, w) if v < w else (w, v)]
                status[w][1] = distance_from_root + 1.0/weight
                stack.append(w)


@profile
def extract_hinge_nodes(status, nodes_sign, with_distances=False):
    """Get nodes which are revealed or have at least 3 incident hinge lines,
    sorted by distance."""
    res = {}
    for v, (_, distance_from_root, nb_incident_hinge_lines, _) in iteritems(status):
        if v in nodes_sign or nb_incident_hinge_lines >= 3:
            res[v] = distance_from_root
    if with_distances:
        return res
    return [_[0] for _ in sorted(iteritems(res), key=lambda x: (x[1], x[0]))]


@profile
def clean_root_hinge(root, tree_adj, status):
    """Remove the single incorrect hinge line starting at the root.

    Done by going down it until we find a node lying on a real hinge line (i.e.
    with 2 incident hinge edges besides the erroneous one from the root)
    """
    node = root
    seen = set()
    while True:
        marked = []
        for child in tree_adj[node]:
            if child in seen:
                continue
            if status[child][3]:
                marked.append(child)
        seen.add(node)
        status[node][2] -= 1
        if len(marked) == 1:
            v = marked[0]
            status[node][3] = False
            node = v
        else:
            return


@profile
def update_mark_status(v, tree_adj, status, v_distance_from_root):
    child_marks = 0
    for neighbor in tree_adj[v]:
        if status[neighbor][1] < v_distance_from_root:
            # that's v parent, as it's closer to root
            continue
        child_marked = status[neighbor][3]
        if child_marked:
            child_marks += 1
            status[neighbor][2] += 1
    status[v][2] = child_marks
    status[v][3] = status[v][3] or child_marks > 0


@profile
def predict_one_node(node, tree_adj, edge_weight, node_signs):
    if len(node_signs) <= 1:
        return -1
    for u in find_hinge_nodes(tree_adj, edge_weight, node_signs, node):
        val = flep(tree_adj, node_signs, edge_weight, u)[0]
        if abs(val) > 1e-4:
            return 1 if val > 0 else -1
    return -1


@profile
def new_online_shazoo(tree_adj, nodes_status, edge_weight, hinge_lines, node_signs, gold_sign):
    """Predict all the signs of `gold_sign` in an online fashion."""
    order = list(gold_sign.keys())
    random.shuffle(order)
    allpred = {}
    for node in order:
        pred = predict_one_node(node, tree_adj, edge_weight, node_signs)
        allpred[node] = pred
        node_signs[node] = gold_sign[node]
    mistakes = sum((1 for n, p in allpred.items() if p != gold_sign[n]))
    print('mistakes: {}'.format(mistakes))
    return order


def assign_gamma(tree_adj, root, ew, parents, node_signs, faulty_sign, only_faulty=True):
    gammas = {root: 1}
    if not tree_adj:
        if only_faulty:
            return gammas if node_signs[root] == faulty_sign else {}
        return gammas
    q = deque()
    q.append(root)
    sum_of_weights = defaultdict(int)
    while q:
        v = q.popleft()
        p = parents[v]
        if p is not None:
            w = ew[(p, v) if p < v else (v, p)]
            gamma = gammas[p]*(w/sum_of_weights[p])
            if v not in node_signs:
                gammas[v] = gamma
            else:
                gammas[v] = gamma if node_signs[v] == faulty_sign else 0
        for u in tree_adj[v]:
            if u == parents[v]:
                continue
            w = ew[(u, v) if u < v else (v, u)]
            sum_of_weights[v] += float(w)
            q.append(u)
    if only_faulty:
        return {u: gammas[u] for u, s in iteritems(node_signs) if s == faulty_sign}
    return gammas


def build_border_tree_from_mincut_run(status, edge_weight):
    parents, leaves_sign, root = {}, {}, None
    for k, v in iteritems(status):
        if v[1] >= 0:
            parents[k] = v[1]
        else:
            parents[k] = None
            root = k
        if MAX_WEIGHT in v[2:]:
            leaves_sign[k] = 1 if v[2] == MAX_WEIGHT else -1

    E, El = set(), []
    tree_adj = {}
    for u in leaves_sign:
        p = parents[u]
        if p is None:
            assert len(leaves_sign) == 1, 'root is a leaf but not the only one!'
            continue
        e = (u, p) if u < p else(p, u)
        w = edge_weight[e]
        El.append((e[0], e[1], w))
        add_edge(tree_adj, *e)
        while p is not None:
            prev = p
            p = parents[prev]
            if p is None:
                break
            e = (prev, p) if prev < p else (p, prev)
            if e in E:
                break
            w = edge_weight[e]
            E.add((e[0], e[1], w))
            add_edge(tree_adj, *e)
    return tree_adj, list(E), El, leaves_sign, parents, root


def predict_one_node_three_methods(node, tree_adj, edge_weight, node_vals):
    node_signs, node_gammas, gamma_signs = node_vals
    if len(node_signs) <= 1:  # don't even bother
        return (-1, None)
    predictions = dict(shazoo=(None, None), rta=(None, None), l2cost=(None, None))
    hinge_nodes = find_hinge_nodes(tree_adj, edge_weight, node_signs, node)
    for u in hinge_nodes:
        status = None
        if predictions['shazoo'][0] is None:
            val, _, status = flep(tree_adj, node_signs, edge_weight, u)
            if abs(val) > 1e-5:
                predictions['shazoo'] = (1 if val > 0 else -1, None)
        if predictions['rta'][0] is None:
            val, _, status = flep(tree_adj, gamma_signs, edge_weight, u)
            if abs(val) > 1e-5:
                border_tree = build_border_tree_from_mincut_run(status, edge_weight)
                predictions['rta'] = (1 if val > 0 else -1, border_tree)
        if predictions['l2cost'][0] is None:
            if status is None:
                _, _, status = flep(tree_adj, node_signs, edge_weight, u)
            border_tree = build_border_tree_from_mincut_run(status, edge_weight)
            _, E, El, leaves_sign, _, _ = border_tree
            L = {u: node_gammas[u] for u in leaves_sign}
            mapped_E, mapped_El_L, mapping = preprocess_edge_and_leaves(E, El, L)
            val = solve_by_zeroing_derivative(mapped_E, mapped_El_L, mapping, L,
                                              reorder=False)[0][u]
            if abs(val) > 1e-5:
                predictions['l2cost'] = (1 if val > 0 else -1, border_tree)
        if all((pred is not None for pred, _ in predictions.values())):
            return predictions
    for method, (pred, tree) in iteritems(predictions):
        if pred is None:
            predictions[method] = (-1, tree)
    return predictions


def sgn(x):
    if abs(x) < 1e-7:
        return 0
    return 1 if x > 0 else -1


def threeway_batch_shazoo(tree_adj, edge_weight, node_signs, gold_sign,
                          order=None):
    """Predict all the signs of `gold_sign` in an online fashion."""
    diff = 0
    if order is None:
        order = list(set(gold_sign.keys()) - set(node_signs.keys()))
        random.shuffle(order)
    allpred = {'shazoo': {}, 'rta': {}, 'l2cost': {}}
    gammas_l2, gammas_rta, gamma_signs = dict(node_signs), dict(node_signs), dict(node_signs)
    node_vals = (node_signs, gammas_l2, gamma_signs)
    for node in order:
        predictions = predict_one_node_three_methods(node, tree_adj, edge_weight, node_vals)
        gold = gold_sign[node]
        node_signs[node] = gold
        gammas_l2[node] = gold
        gammas_rta[node] = gold
        gamma_signs[node] = gold
        if not isinstance(predictions, dict):
            allpred['shazoo'][node] = -1
            allpred['rta'][node] = -1
            allpred['l2cost'][node] = -1
            continue
        if predictions['rta'][0] != predictions['l2cost'][0]:
            diff += 1
        for method in ['shazoo', 'rta', 'l2cost']:
            pred = predictions[method][0]
            allpred[method][node] = pred
            if predictions[method][1] is None:
                continue
            if gold != pred and method == 'rta':
                border_tree_adj, _, _, leaves_sign, parents, root = predictions[method][1]
                update = assign_gamma(border_tree_adj, root, edge_weight, parents, leaves_sign, -gold)
                for leaf, gamma in iteritems(update):
                    assert gamma > 0
                    gammas_rta[leaf] += gold*gamma
                    gamma_signs[leaf] = sgn(gammas_rta[leaf])
            if gold != pred and method == 'l2cost':
                border_tree_adj, _, _, leaves_sign, parents, root = predictions[method][1]
                update = assign_gamma(border_tree_adj, root, edge_weight, parents, leaves_sign, -gold)
                for leaf, gamma in iteritems(update):
                    gammas_l2[leaf] += gold*gamma
    gold, pred = [], {'shazoo': [], 'rta': [], 'l2cost': []}
    for node in sorted(order):
        pred['shazoo'].append(allpred['shazoo'][node])
        pred['rta'].append(allpred['rta'][node])
        pred['l2cost'].append(allpred['l2cost'][node])
        gold.append(node_signs[node])
    print(diff)
    return gold, pred


def preprocess_edge_and_leaves(E, El, L):
    E = sorted(E)
    El = sorted(El)
    mapping = {v: i for i, v in enumerate(sorted({u for e in E+El for u in e[:2]}))}
    mapped_E = np.array([[mapping[u], mapping[v], w] for u, v, w in E])
    mapped_El_L = np.zeros((len(El), 3))
    for i, (u, v, w) in enumerate(El):
        if u in L:
            u, leaf = v, u
        else:
            u, leaf = u, v
        mapped_El_L[i, :] = (mapping[u], L[leaf], w)
    return mapped_E, mapped_El_L, mapping


def np_cost_l2(x, mapped_E, mapped_El_L):
    if mapped_E.size > 0:
        internal = np.sum(mapped_E[:, 2]*((x[mapped_E[:, 0].astype(int)] - x[mapped_E[:, 1].astype(int)])**2))
    else:
        internal = 0
    border = np.sum(mapped_El_L[:, 2]*((x[mapped_El_L[:, 0].astype(int)] - mapped_El_L[:, 1])**2))
    return internal + border


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


if __name__ == '__main__':
    # pylint: disable=C0103
    import sys
    import persistent as p
    random.seed(123459)

    num_rep = 10
    res = np.zeros((num_rep, 6))
    for i in range(num_rep):
        gr, phi_edges = make_graph(1000)
        # start = clock()
        # shazoo(*gr)
        # print(clock() - start)
        from collections import Counter
        count = Counter(gr[-1].values())
        res[i, :2] = phi_edges, count[1]
        # start = clock()
        # order = new_online_shazoo(gr[0], None, gr[2], None, {}, gr[-1])
        # print(clock() - start)
        order = None
        start = clock()
        gold, preds = threeway_batch_shazoo(gr[0], gr[2], {}, gr[-1], order)
        for j, (method, pred) in enumerate(sorted(preds.items(), key=lambda x: x[0])):
            mistakes = sum((1 for g, p in zip(gold, pred) if p != g))
            print('{} made {} mistakes'.format(method.ljust(6), mistakes))
            res[i, 2+j] = mistakes
        time_elapsed = (clock() - start)
        res[i, -1] = time_elapsed
        print(time_elapsed)
        np.savez_compressed('shazoo_run2', res=res, do_compression=True)
    sys.exit()
    start = clock()
    shazoo(*make_graph(4000))
    print(clock() - start)

    adj, _, ew, _, _, gold_sign = make_graph(400)
    train_vertices = random.sample(gold_sign.keys(), 70)
    gold, pred = offline_shazoo(adj, ew, gold_sign, train_vertices)
    print(sum((1 for g, p in zip(gold, pred) if g != p)))

    timing = []
    for i in range(8):
        del FLEP_CALLS_TIMING[:]
        start = clock()
        shazoo(*make_graph(3250))
        p.save_var('flep_{}.my'.format(i), FLEP_CALLS_TIMING)
        # print('done in {:.3f} sec'.format(clock() - start))
        timing.append(clock() - start)
    print('avrg run: {:.3f}'.format(sum(timing)/len(timing)))

    def run_once(size):
        cexp.fast_preferential_attachment(size, 1)
        adj = cexp.redensify.G
        ew = {e: 120*random.random() for e in cexp.redensify.EDGES_SIGN}
        ns = {n: random.random() > .5 for n in adj
              if len(adj[n]) == 1 and random.random() < .7}
        root = max(adj.items(), key=lambda x: len(x[1]))[0]
        flep(adj, ns, ew, root)
    run_once(1000)
    run_once(1000)
    start = clock()
    run_once(200000)
    print('done in {:.3f} sec'.format(clock() - start))
