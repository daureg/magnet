"""
Implement the Shazoo node binary classification algorithm.

from Vitale, F., Cesa-Bianchi, N., Gentile, C., & Zappella, G. (2011).
See the tree through the lines: the Shazoo algorithm.
In Advances in Neural Information Processing Systems 24 (pp. 1584–1592).
http://papers.nips.cc/paper/4476-see-the-tree-through-the-lines-the-shazoo-algorithm .
"""
from collections import deque
import convert_experiment as cexp
import random
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
def flep(tree_adj, nodes_sign, edge_weight, root):
    """Compute the sign of the `root` that yield the smallest weighted cut in
    `tree_adj` given the already revealed `nodes_sign`."""
    start = clock()
    assert isinstance(tree_adj, dict)
    if root in nodes_sign:
        return nodes_sign[root]
    assert root not in nodes_sign
    stack = []
    status = {_: (False, -1, 0, 0) for _ in tree_adj}
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
                intermediate = {n: (vals[2], vals[3])
                                for n, vals in status.items()
                                if vals[0] and n not in nodes_sign}
                FLEP_CALLS_TIMING.append(clock() - start)
                return (cutn - cutp, intermediate)

        if not discovered:
            status[v] = (True, pred, cutp, cutn)
            if v in nodes_sign:
                # don't go beyond revealed nodes
                continue
            stack.append(-(v+100))
            for w in tree_adj[v]:
                discovered, pred, cutp, cutn = status[w]
                if pred == -1:
                    if w in nodes_sign:
                        cutp, cutn = {-1: (MAX_WEIGHT, 0),
                                      1: (0, MAX_WEIGHT)}[nodes_sign[w]]
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
    status = {u: (False, 0) for u in tree_adj}
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
    return (adj, nodes_status, ew, hinge_lines, nodes_sign, gold_sign)


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
    _, rooted_cut = flep(tree_adj, nodes_sign, edge_weight, root)
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


if __name__ == '__main__':
    # pylint: disable=C0103
    import sys
    import persistent as p
    random.seed(123456)

    for i in range(10):
        shazoo(*make_graph(800))
    sys.exit()

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
