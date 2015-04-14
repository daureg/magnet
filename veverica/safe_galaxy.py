from collections import deque, defaultdict, Counter
from itertools import combinations
from copy import deepcopy
import galaxy as gx
from timeit import default_timer as clock
UNKNOWN, QUERIED, PREDICTED = 0, 1, 2


def safe_bfs_search(adjacency, edge_signs, edge_status, nodes_subset, src,
                    dst):
    """Look for a path between `src` and `dst` in the `nodes_subset` of
    `adjacency` using only queried edges. Moreover, the past must not contain
    more than one negative edge and it returns this number (or None if there's
    no such path."""
    assert isinstance(nodes_subset, set)
    if src > dst:
        src, dst = dst, src
    border = deque()
    discovered = {k: (False, 0) for k in nodes_subset}
    border.append(src)
    discovered[src] = (True, 0)
    found = False
    while border and not found::
        v = border.popleft()
        negativity = discovered[v][1]
        for w in adjacency[v].intersection(nodes_subset):
            if not discovered[w][0]:
                e = (v, w) if v < w else (w, v)
                if edge_status[e] != QUERIED:
                    continue
                sign = edge_signs[e]
                w_negativity = negativity + {False: 1, True: 0}[sign]
                if w_negativity <= 1:
                    discovered[w] = (True, w_negativity)
                    border.append(w)
                    if w == dst:
                        found = True
                        break
    if not found:
        return None
    assert discovered[dst][0]
    assert 0 <= discovered[dst][1] <= 1
    return discovered[dst][1]


def safely_get_sign(adjacency, edge, nodes, edge_status, edge_signs):
    """Try to predict the sign of `edge` by finding a path in `nodes`"""
    prediction = safe_bfs_search(adjacency, edge_signs, edge_status, nodes,
                                 *edge)
    if prediction is not None:
        edge_status[edge] = PREDICTED
        return True, 1 if prediction == 0 else -1
    # If we can't predict, we query the edge (which in practice only means we
    # can now look at its sign)
    edge_status[edge] = QUERIED
    return False, None


# The two following function are a bit awkward, since in first level they deal
# with stars of nodes and after that with set of nodes (without a single center
# in the original graph
def all_edges_within(adjacency, star, edge_status):
    """return the set of all test edges within `star`"""
    edges = []
    nodes_subset = set(star if not hasattr(star, 'points') else star.points)
    for u in nodes_subset:
        for v in adjacency[u].intersection(nodes_subset):
            e = (u, v) if u < v else (v, u)
            if edge_status[e] == UNKNOWN:
                edges.append(e)
    return set(edges)


def all_edges_between(adjacency, s1, s2, edge_status):
    """return the set of all test edges of `G` between two stars"""
    if hasattr(s1, 'center'):
        size1 = len(adjacency[s1.center])
        size2 = len(adjacency[s2.center])
    else:
        size1, size2 = len(s1), len(s2)
    if size1 > size2:
        s1, s2 = s2, s1
    edges = []
    if hasattr(s1, 'center'):
        srcs = s1.points
        dests = set([s2.center] + s2.points)
    else:
        srcs, dests = s1, s2
    for u in srcs:
        for v in adjacency[u].intersection(dests):
            edge = (u, v) if u < v else (v, u)
            if edge_status[edge] == UNKNOWN:
                edges.append(edge)
    return set(edges)


class id_dict(dict):
    """Identity dictionary"""

    def __getitem__(self, key):
        """return the key itself as value"""
        return key


def mark_queried_edges(edge_status, edges_to_mark, edge_mappings):
    """Update `edge_status` by translating to initial level `edges_to_mark`"""
    if len(edge_mappings) == 0:
        trad = id_dict()
    else:
        trad = deepcopy(edge_mappings[-1])
        for mapping in reversed(edge_mappings[:-1]):
            lower_trad = {}
            for high_edge, low_level_edge in trad.items():
                lower_trad[high_edge] = mapping[low_level_edge]
            trad = lower_trad
    for e in edges_to_mark:
        edge_status[trad[e]] = QUERIED


def safe_galaxy_maker(G, k, edge_signs):
    """Galaxy maker, but predicting only if there are paths with no more than
    one negative edge, querying otherwise."""
    current_graph, ems, _ = G, [], []
    star_membership = {node: node for node in G.keys()}
    edge_status = {edge: UNKNOWN for edge in edge_signs.keys()}
    original_basis = id_dict()
    gold, preds = [], []

    def predict_edges_batch(test_edges, nodes):
        if not test_edges:  # common fast path
            return
        _nodes = nodes
        if hasattr(nodes, 'center'):
            _nodes = set([nodes.center] + nodes.points)
        for e in test_edges:
            pred, sign = safely_get_sign(G, e, _nodes, edge_status, edge_signs)
            if pred:
                preds.append(sign)
                gold.append(1 if edge_signs[e] else -1)

    for i in range(k):
        start = clock()
        stars, stars_edges = gx.stars_maker(current_graph)
        flat_stars_edges = (e for s in stars_edges for e in s)
        mark_queried_edges(edge_status, flat_stars_edges, ems)
        for s in stars:
            if i == 0:
                to_be_predicted = all_edges_within(G, s, edge_status)
                # TODO instead of querying, should I put failed edges on hold,
                # hoping that future successful prediction will yield safe
                # path for them? If the to_be_predicted does not change
                # size after one pass, then I query everything.
                # It could be a function taking to_be_predicted and node
                # subset. This would require modifying safely_get_sign to not
                # query edges.
                # In fact, I still don't get the idea: if I can't predict the
                # edge and don't query it, nothing will change since
                # safe_bfs_search don't use predicted edge. So I can make as
                # many pass as I want, it won't change anything
                predict_edges_batch(to_be_predicted, s)
                continue
            center = original_basis[s.center]
            radials = [original_basis[p] for p in s.points]
            # TODO Ask Fabio about this all points thing
            allpts = {p for c in [center]+radials for p in c}
            for component in ([center] + radials):
                to_be_predicted = all_edges_within(G, component, edge_status)
                predict_edges_batch(to_be_predicted, component)
            for component in radials:
                to_be_predicted = all_edges_between(G, component, center,
                                                    edge_status)
                predict_edges_batch(to_be_predicted, allpts)
                # component | center)
            for comp1, comp2 in combinations(radials, 2):
                to_be_predicted = all_edges_between(G, comp1, comp2,
                                                    edge_status)
                predict_edges_batch(to_be_predicted, allpts)
                # comp1.union(comp2).union(center))
        collapsed_graph, _, em, sm = gx.collapse_stars(current_graph, stars)
        if i == 0:
            star_membership = sm
        else:
            for orig_nodes, previous_star in star_membership.items():
                star_membership[orig_nodes] = sm[previous_star]
        original_basis = defaultdict(set)
        for orig_id, star_id in star_membership.items():
            original_basis[star_id].add(orig_id)
        duration = clock() - start
        print('iteration {} in {:.3f} seconds'.format(str(i).ljust(3),
                                                      duration))
        print(i, Counter(edge_status.values()))
        ems.append(em)
        if len(em) == 0 or len(collapsed_graph) == len(current_graph):
            break
        current_graph = collapsed_graph
    return gold, preds, edge_status


def save_edges(edge_status, outname):
    edges = [e for e, kind in edge_status.items()
             if kind == QUERIED]
    with open(outname+'.edges', 'w') as f:
        f.write('\n'.join(('{}, {}'.format(*e) for e in edges)))
