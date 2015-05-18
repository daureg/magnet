#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Implement the Shazoo node binary classification algorithm from
Vitale, F., Cesa-Bianchi, N., Gentile, C., & Zappella, G. (2011).
See the tree through the lines: the Shazoo algorithm.
In Advances in Neural Information Processing Systems 24 (pp. 1584–1592).
http://papers.nips.cc/paper/4476-see-the-tree-through-the-lines-the-shazoo-algorithm
"""
from collections import deque


MAX_WEIGHT = int(2e9)
UNKNOWN, REVEALED, FORK, HINGE = 0, 1, 2, 3


def _edge(u, v):
    """reorder u and v"""
    return (u, v) if u < v else (v, u)


def flep(tree_adj, nodes_sign, edge_weight, root):
    """Compute the sign of the `root` that yield the smallest weighted cut in
    `tree_adj` given the already revealed `nodes_sign`."""
    assert isinstance(tree_adj, dict)
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
            children = (u for u in tree_adj[v] if status[u][1] == v)
            for child in children:
                eweight = edge_weight[_edge(child, v)]
                _, _, childp, childn = status[child]
                cutp += min(childp, childn + eweight)
                cutn += min(childn, childp + eweight)
            status[v] = (discovered, pred, cutp, cutn)
            # print('{}: (+: {}, -: {})'.format(v, cutp, cutn))
            if v == root:
                # return {n: vals[3] - vals[2] for n, vals in status.items()}
                return cutn - cutp

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


def is_a_fork(tree_adj, node, hinge_lines):
    """If node has more than 3 hinge edges incident, it's a fork"""
    incident_hinge = 0
    for u in tree_adj[node]:
        incident_hinge += int(hinge_lines[_edge(u, node)])
        if incident_hinge >= 3:
            return True
    return False


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
                nodes_status[potential_fork] = FORK
        if nodes_status[parent] in [REVEALED, FORK]:
            break
        node, parent = parent, ancestors[parent]


def predict_node_sign(tree_adj, node, nodes_status, nodes_sign, hinge_lines,
                      edge_weight):
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
        if v_status == FORK:
            estim = flep(tree_adj, nodes_sign, edge_weight, v)
            if abs(estim) > 1e-4:
                connect_nodes[v] = 1 if estim > 0 else -1
                if distance_from_root < min_connect_distance:
                    min_connect, min_connect_distance = v, distance_from_root
        if distance_from_root >= min_connect_distance:
            continue
        for w in tree_adj[v]:
            edge = _edge(v, w)
            if not status[w][0]:
                q.append(w)
                status[w] = (True, distance_from_root + 1/edge_weight[edge])

    # print(connect_nodes, min_connect, min_connect_distance)
    return -1 if min_connect is None else connect_nodes[min_connect]


if __name__ == '__main__':
    # pylint: disable=C0103
    import convert_experiment as cexp
    import random
    from timeit import default_timer as clock

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
