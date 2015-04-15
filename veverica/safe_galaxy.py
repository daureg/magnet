"""Implement variants of the original GalaxyMaker algorithm"""
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
    while border and not found:
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


def all_edges_between(adjacency, s1, s2, edge_status=None):
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
        srcs = set([s1.center] + s1.points)
        dests = set([s2.center] + s2.points)
    else:
        srcs, dests = s1, s2
    for u in srcs:
        for v in adjacency[u].intersection(dests):
            edge = (u, v) if u < v else (v, u)
            if edge_status is None or edge_status[edge] == UNKNOWN:
                edges.append(edge)
    return set(edges)


class IdentityDict(dict):
    """Identity dictionary"""

    def __getitem__(self, key):
        """return the key itself as value"""
        return key


def mark_queried_edges(edge_status, edges_to_mark, edge_mappings):
    """Update `edge_status` by translating to initial level `edges_to_mark`"""
    if len(edge_mappings) == 0:
        trad = IdentityDict()
    else:
        trad = deepcopy(edge_mappings[-1])
        for mapping in reversed(edge_mappings[:-1]):
            lower_trad = {}
            for high_edge, low_level_edge in trad.items():
                lower_trad[high_edge] = mapping[low_level_edge]
            trad = lower_trad
    for e in edges_to_mark:
        edge_status[trad[e]] = QUERIED


def save_edges(edge_status, outname):
    """Write the list of queried edges in `outname`."""
    edges = [e for e, kind in edge_status.items()
             if kind == QUERIED]
    with open(outname+'.edges', 'w') as f:
        f.write('\n'.join(('{}, {}'.format(*e) for e in edges)))


def meta_galaxy(graph, edge_signs, nb_iter, outname, safe=False, short=False):
    """General structure of the galaxy maker algorithm with two optional
    variants:
    -safe: predict only paths with less than 2 negative edges, query otherwise
    -short: create link between stars based on nodes centrality"""
    current_graph, ems, all_stars = graph, [], []
    star_membership = {node: node for node in graph.keys()}
    centrality = {node: 0 for node in graph.keys()} if short else None
    edge_status = None
    if safe:
        edge_status = {edge: UNKNOWN for edge in edge_signs.keys()}
    original_basis = IdentityDict()
    gold, preds = [], []

    for i in range(nb_iter):
        start = clock()
        first_iter = i == 0
        stars, stars_edges = gx.stars_maker(current_graph)
        all_stars.append(stars_edges)
        if safe:
            flat_stars_edges = (e for s in stars_edges for e in s)
            mark_queried_edges(edge_status, flat_stars_edges, ems)
        graph_info = (graph, edge_status, edge_signs, original_basis)
        labels = (gold, preds)
        for_each_stars(stars, centrality, edge_status, first_iter, graph_info,
                       labels)

        collapsed_graph, em, sm = collapse_stars(current_graph, stars,
                                                 centrality, edge_status)
        long_res = update_nodes_mapping(star_membership, sm, first_iter)
        star_membership, original_basis = long_res
        duration = clock() - start
        print('iteration {} in {:.3f} seconds'.format(str(i).ljust(3),
                                                      duration))
        if safe:
            print(Counter(edge_status.values()))
        ems.append(em)
        if outname:
            filename = '{}_{}'.format(outname, i)
            if safe:
                save_edges(edge_status, filename)
            else:
                gx.export_spanner(all_stars, ems, star_membership, filename)
        if len(em) == 0 or len(collapsed_graph) == len(current_graph):
            break
        current_graph = collapsed_graph
    return None if not safe else (gold, preds, edge_status)


def for_each_stars(stars, centrality, edge_status, first_iter, graph_info,
                   labels):
    if not (centrality or edge_status):
        return
    G, edge_status, edge_signs, original_basis = graph_info
    for s in stars:
        if edge_status and first_iter:
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
            predict_edges_batch(to_be_predicted, s, graph_info, labels)
        center = original_basis[s.center]
        radials = [original_basis[p] for p in s.points]
        if centrality:
            for component in radials:
                if isinstance(component, int):
                    assert first_iter
                    centrality[component] += 1
                else:
                    for point in component:
                        centrality[point] += 1
        if edge_status is None or first_iter:
            continue
        # TODO Ask Fabio about this all points thing
        allpts = {p for c in [center]+radials for p in c}
        for component in ([center] + radials):
            to_be_predicted = all_edges_within(G, component, edge_status)
            predict_edges_batch(to_be_predicted, component, graph_info, labels)
        for component in radials:
            to_be_predicted = all_edges_between(G, component, center,
                                                edge_status)
            predict_edges_batch(to_be_predicted, allpts, graph_info, labels)
            # component | center)
        for comp1, comp2 in combinations(radials, 2):
            to_be_predicted = all_edges_between(G, comp1, comp2,
                                                edge_status)
            predict_edges_batch(to_be_predicted, allpts, graph_info, labels)
            # comp1.union(comp2).union(center))


def update_nodes_mapping(star_membership, new_sm, first_iter):
    """After a collapse step, update the mapping between node indices in the
    new graph and original one."""
    if first_iter:
        star_membership = new_sm
    else:
        for orig_nodes, previous_star in star_membership.items():
            star_membership[orig_nodes] = new_sm[previous_star]
    original_basis = defaultdict(set)
    for orig_id, star_id in star_membership.items():
        original_basis[star_id].add(orig_id)
    return star_membership, original_basis


def collapse_stars(G, stars, centrality, edge_status):
    """From a graph `G` and its list of `stars`, return a new graph with a
    node for each star and a link between them if they are connected in the
    previous level graph.
    If centrality is not None, create the link with highest centrality.
    e_mapping: edge in G' -> edge in G
    """
    Gprime = {_: set() for _ in range(len(stars))}
    # at the end, each node of G will be key, whose value is the star it
    # belongs to
    star_membership = {}
    e_mapping = {}
    for i, s in enumerate(stars):
        star_membership[s.center] = i
        for p in s.points:
            star_membership[p] = i
        for j, t in enumerate(stars[i+1:]):
            if centrality is None:
                min_edge = gx.edge_between(G, s, t)
            else:
                between = all_edges_between(G, s, t)
                min_val, min_edge = 1000, None
                for u, v in between:
                    val = centrality[u] + centrality[v]
                    if val < min_val:
                        min_val, min_edge = val, (u, v)
            if min_edge:
                Gprime[i].add(j+i+1)
                Gprime[j+i+1].add(i)
                e_mapping[(i, i+1+j)] = min_edge
    return Gprime, e_mapping, star_membership


def predict_edges_batch(test_edges, nodes, graph_info, labels):
    """update `labels` with prediction over `test_edges` by searching for safe
    paths within `nodes`"""
    if not test_edges:  # common fast path
        return
    graph, edge_status, edge_signs, _ = graph_info
    gold, preds = labels
    _nodes = nodes
    if hasattr(nodes, 'center'):
        _nodes = set([nodes.center] + nodes.points)
    for e in test_edges:
        pred, sign = safely_get_sign(graph, e, _nodes, edge_status, edge_signs)
        if pred:
            preds.append(sign)
            gold.append(1 if edge_signs[e] else -1)

if __name__ == '__main__':
    # pylint: disable=C0103
    import args_experiments as ae
    import convert_experiment as cexp
    import real_world as rw
    import redensify
    parser = ae.get_parser('Compute a galaxy tree')
    args = parser.parse_args()
    a = ae.further_parsing(args)
    basename, seeds, synthetic_data, prefix, noise, balanced = a

    if synthetic_data:
        import graph_tool as gt
        g = gt.load_graph(basename+'.gt')
        cexp.to_python_graph(g)
    else:
        rw.read_original_graph(basename, seed=args.seed, balanced=balanced)
        redensify.G = deepcopy(rw.G)
        redensify.EDGES_SIGN = deepcopy(rw.EDGE_SIGN)

    suffixes = ('_bal' if args.balanced else '',
                '_short' if args.short else '',
                '_safe' if args.safe else '')
    outname = 'universe/{}{}{}{}_test'.format(args.data.lower(), *suffixes)
    print(outname)
    res = meta_galaxy(redensify.G, redensify.EDGES_SIGN, 10, outname,
                      safe=args.safe, short=args.short)
    if args.safe:
        gold, pred, _ = res
        import persistent
        persistent.save_var(outname+'_res.my', (gold, pred))
