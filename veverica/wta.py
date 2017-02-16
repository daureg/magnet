# vim: set fileencoding=utf-8
"""
Implement the WTA node binary classification algorithm.

from Cesa-Bianchi, N., Gentile, C., Vitale, F. & Zappella, G.
Random Spanning Trees and the Prediction of Weighted Graphs.
J. Mach. Learn. Res. 14, 1251–1284 (2013).
http://www.jmlr.org/papers/v14/cesa-bianchi13a.html .
"""
from __future__ import division
from collections import defaultdict
from attr.validators import instance_of
import attr


def is_binary(instance, attribute, x):
    if x not in {-1, 1, None}:
        msg = 'Expecting a binary value for {} instead of {}'
        raise ValueError(msg.format(attribute.name, x))


def is_line_node(instance, attribute, x):
    if not isinstance(x, LineNode) and x is not None:
        msg = 'Expecting a LineNode for {} instead of {}'
        raise ValueError(msg.format(attribute.name, type(x)))


@attr.s(slots=True)
class LineNode(object):
    id_ = attr.ib(validator=instance_of(int))
    left = attr.ib(default=None, cmp=False, repr=False, validator=is_line_node)
    right = attr.ib(default=None, cmp=False, repr=False, validator=is_line_node)
    sign = attr.ib(default=None, validator=is_binary)
    smallest_dst = attr.ib(default=1e9, validator=instance_of(float))
    source = attr.ib(default=-1, validator=instance_of(int))
    pred_sign = attr.ib(default=None, validator=is_binary)


@attr.s(slots=True)
class Propagated(object):
    id_ = attr.ib(validator=instance_of(int))
    sign = attr.ib(validator=is_binary)
    origin = attr.ib(validator=instance_of(int))
    going_right = attr.ib(validator=instance_of(bool))
    pos = attr.ib(cmp=False, hash=False, validator=instance_of(int))
    dst = attr.ib(cmp=False, hash=False, convert=float, validator=instance_of(float))


def propagate_revelead_signs(nodes_line, edge_weight, return_prediction=True):
    propas = set()
    for i, u in enumerate(nodes_line):
        if u.sign is not None:
            j = len(propas)//2
            propas.add(Propagated(id_=2*j, pos=i, dst=0, sign=u.sign, going_right=False, origin=i))
            propas.add(Propagated(id_=2*j+1, pos=i, dst=0, sign=u.sign, going_right=True, origin=i))
            u.source = i
            u.smallest_dst = 0
    n = len(nodes_line)
    nb_iter = 0
    while propas and nb_iter < 2*n:
        to_remove = set()
        for p in iter(propas):
            nei_idx = p.pos + (1 if p.going_right else -1)
            if not 0 <= nei_idx < n:
                to_remove.add(p)
                continue
            nei = nodes_line[nei_idx]
            u, v = nei.id_, nodes_line[p.pos].id_
            w = edge_weight[(u, v) if u < v else (v, u)]
            dst = p.dst + 1/w
            if nei.sign is not None or nei.smallest_dst < dst + 1e-9:
                to_remove.add(p)
                continue
            nei.pred_sign = p.sign
            nei.smallest_dst = dst
            nei.source = p.origin
            p.pos = nei_idx
            p.dst += 1/w
        propas.difference_update(to_remove)
        nb_iter += 1
    if return_prediction:
        return {u.id_: u.pred_sign for u in nodes_line if u.pred_sign is not None}


def dfs_order(tree_adj, edge_weight, root):
    assert isinstance(tree_adj, dict)
    stack = [root, ]
    discovered = defaultdict(bool)
    order, seen = [], set()
    while stack:
        v = stack.pop()
        order.append(v)
        seen.add(v)
        if len(seen) == len(tree_adj):
            return order
        if not discovered[v]:
            discovered[v] = True
            # TODO: I use sorted here for testing but in practice I could get more variety
            for w in sorted(tree_adj[v], reverse=True):
                if not discovered[w]:
                    stack.append(v)
                    stack.append(w)


def remove_duplicates(line, ew):
    seen = set((line[0], ))
    i, n = 0, len(line)
    res = []
    while i < n:
        u = line[i]
        weights = set()
        for j, v in enumerate(line[i+1:]):
            prev_u = line[i+1+j-1]
            weights.add(ew[(prev_u, v)] if prev_u < v else ew[(v, prev_u)])
            if v not in seen:
                break
        w = None if not weights else min(weights)
        seen.add(v)
        res.append((u, w))
        i += j+1
    return res


def linearize_tree(tree_adj, edge_weight, root):
    ordered_node = dfs_order(tree_adj, edge_weight, root)
    return remove_duplicates(ordered_node, edge_weight)


def convert_to_line(node_and_weight):
    nodes = []
    line_weight = {}
    for (u, w), (v, _) in zip(node_and_weight, node_and_weight[1:]):
        nodes.append(LineNode(id_=u))
        line_weight[(u, v) if u < v else (v, u)] = w
    nodes.append(LineNode(id_=node_and_weight[-1][0]))
    for u, v in zip(nodes, nodes[1:]):
        u.right = v
        v.left = u
    return nodes, line_weight


def predict_signs(nodes_line, edge_weight, training_set):
    for n in nodes_line:
        n.pred_sign = None
        n.sign = training_set.get(n.id_, None)
        n.smallest_dst = 1e9
        n.source = -1
    return propagate_revelead_signs(nodes_line, edge_weight)


if __name__ == "__main__":
    import random
    from shazoo_exps import make_graph
    from timeit import default_timer as clock
    n = 1000
    random.seed(123)
    tree_adj, ew, signs, _ = make_graph(20000)
    timings = []
    nodes_line, edge_weight = convert_to_line(linearize_tree(tree_adj, ew, 0))
    nrep = 20
    for _ in range(nrep):
        training_set = {u: signs[u] for u in random.sample(list(range(n)), int(.1*n))}
        start = clock()
        predict_signs(nodes_line, edge_weight, training_set)
        timings.append(clock() - start)
    print('\t'.join(('{:.3g}'.format(t) for t in timings)))
    print(sum(timings[1:])/(nrep-1))
